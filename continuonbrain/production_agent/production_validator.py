"""
Production Validation Suite

Comprehensive testing to ensure the seed model is ready for production:
1. Latency requirements
2. Stability under stress
3. Safety behaviors
4. Integration tests
5. Hardware compatibility
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class ValidationLevel(Enum):
    """Validation severity levels."""
    CRITICAL = "critical"   # Must pass for production
    IMPORTANT = "important"  # Should pass for production
    RECOMMENDED = "recommended"  # Nice to have


class ValidationStatus(Enum):
    """Test result status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class ValidationResult:
    """Result of a single validation test."""
    name: str
    level: ValidationLevel
    status: ValidationStatus
    message: str
    metric_value: Optional[float] = None
    metric_target: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: str
    model_path: str
    results: List[ValidationResult] = field(default_factory=list)
    overall_status: ValidationStatus = ValidationStatus.PASSED

    @property
    def critical_passed(self) -> bool:
        return all(
            r.status == ValidationStatus.PASSED
            for r in self.results
            if r.level == ValidationLevel.CRITICAL
        )

    @property
    def all_passed(self) -> bool:
        return all(r.status == ValidationStatus.PASSED for r in self.results)

    @property
    def summary(self) -> Dict[str, int]:
        return {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.status == ValidationStatus.PASSED),
            "failed": sum(1 for r in self.results if r.status == ValidationStatus.FAILED),
            "warning": sum(1 for r in self.results if r.status == ValidationStatus.WARNING),
            "skipped": sum(1 for r in self.results if r.status == ValidationStatus.SKIPPED),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model_path": self.model_path,
            "overall_status": self.overall_status.value,
            "summary": self.summary,
            "critical_passed": self.critical_passed,
            "results": [
                {
                    "name": r.name,
                    "level": r.level.value,
                    "status": r.status.value,
                    "message": r.message,
                    "metric_value": r.metric_value,
                    "metric_target": r.metric_target,
                    "duration_ms": r.duration_ms,
                }
                for r in self.results
            ],
        }


class ProductionValidator:
    """
    Validates seed model for production deployment.

    Checks:
    - Performance (latency, throughput)
    - Stability (no NaN, bounded states)
    - Safety (emergency stop, limits)
    - Integration (chat, action mapping)
    """

    # Target metrics
    TARGET_LATENCY_MS = 100.0
    TARGET_THROUGHPUT_HZ = 10.0
    MAX_STATE_VALUE = 15.0
    MIN_CONFIDENCE = 0.1
    STRESS_TEST_ITERATIONS = 100
    STABILITY_TEST_DURATION_S = 10.0

    def __init__(
        self,
        model_path: Optional[Path] = None,
        verbose: bool = True,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.verbose = verbose
        self.results: List[ValidationResult] = []

        # Model components
        self.model = None
        self.params = None
        self.config = None

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def load_model(self, model_path: Optional[Path] = None) -> bool:
        """Load the seed model for validation."""
        if not JAX_AVAILABLE:
            self._log("JAX not available")
            return False

        path = model_path or self.model_path
        if path is None:
            self._log("No model path specified")
            return False

        path = Path(path)

        try:
            import pickle
            from continuonbrain.jax_models.core_model import CoreModel
            from continuonbrain.jax_models.config import CoreModelConfig

            # Load manifest
            manifest_path = path / "model_manifest.json" if path.is_dir() else path.parent / "model_manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                self.config = CoreModelConfig(**manifest["config"])
                self.obs_dim = manifest.get("input_dims", {}).get("obs_dim", 128)
                self.action_dim = manifest.get("input_dims", {}).get("action_dim", 32)
                self.output_dim = manifest.get("output_dim", 32)
            else:
                from continuonbrain.jax_models.config_presets import get_config_for_preset
                self.config = get_config_for_preset("pi5")
                self.obs_dim = 128
                self.action_dim = 32
                self.output_dim = 32

            # Load params
            params_path = path / "params_step_16.pkl" if path.is_dir() else path
            # Also check for optimized params
            if not params_path.exists() and path.is_dir():
                params_path = path / "optimized_params.pkl"
            if params_path.exists():
                with open(params_path, 'rb') as f:
                    data = pickle.load(f)
                    # Handle nested params structure
                    if 'params' in data:
                        if isinstance(data['params'], dict) and 'params' in data['params']:
                            # Double-nested
                            self.params = data['params']
                        else:
                            self.params = data
                    else:
                        self.params = {'params': data}

            # Create model
            self.model = CoreModel(
                config=self.config,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                output_dim=self.output_dim,
            )

            self._log(f"Model loaded from {path}")
            return True

        except Exception as e:
            self._log(f"Failed to load model: {e}")
            return False

    def _init_state(self) -> Dict[str, Any]:
        """Initialize model state."""
        return {
            's': jnp.zeros((1, self.config.d_s)),
            'w': jnp.zeros((1, self.config.d_w)),
            'p': jnp.zeros((1, self.config.d_p)),
            'cms_memories': [
                jnp.zeros((1, sz, dim))
                for sz, dim in zip(self.config.cms_sizes, self.config.cms_dims)
            ],
            'cms_keys': [
                jnp.zeros((1, sz, self.config.d_k))
                for sz in self.config.cms_sizes
            ],
        }

    def _run_inference(self, obs: jnp.ndarray, state: Dict) -> Tuple[jnp.ndarray, Dict]:
        """Run single inference."""
        action = jnp.zeros((1, self.action_dim))
        reward = jnp.zeros((1, 1))
        output, info = self.model.apply(
            self.params, obs, action, reward,
            state['s'], state['w'], state['p'],
            state['cms_memories'], state['cms_keys'],
        )
        return output, {
            's': info['fast_state'],
            'w': info['wave_state'],
            'p': info['particle_state'],
            'cms_memories': info['cms_memories'],
            'cms_keys': info['cms_keys'],
        }

    # ==================== Validation Tests ====================

    def validate_latency(self) -> ValidationResult:
        """Validate inference latency meets real-time requirements."""
        self._log("\n[Test] Latency Validation")
        start = time.time()

        if self.model is None:
            return ValidationResult(
                name="Latency",
                level=ValidationLevel.CRITICAL,
                status=ValidationStatus.SKIPPED,
                message="Model not loaded",
            )

        state = self._init_state()
        obs = jax.random.normal(jax.random.PRNGKey(0), (1, self.obs_dim))

        # Warmup
        for _ in range(5):
            self._run_inference(obs, state)

        # Measure
        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            self._run_inference(obs, state)
            latencies.append((time.perf_counter() - t0) * 1000)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        passed = avg_latency < self.TARGET_LATENCY_MS

        result = ValidationResult(
            name="Latency",
            level=ValidationLevel.CRITICAL,
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            message=f"Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms (target: <{self.TARGET_LATENCY_MS}ms)",
            metric_value=avg_latency,
            metric_target=self.TARGET_LATENCY_MS,
            details={"p95_ms": p95_latency, "min_ms": min(latencies), "max_ms": max(latencies)},
            duration_ms=(time.time() - start) * 1000,
        )
        self._log(f"    {result.status.value.upper()}: {result.message}")
        return result

    def validate_throughput(self) -> ValidationResult:
        """Validate throughput meets control frequency requirements."""
        self._log("\n[Test] Throughput Validation")
        start = time.time()

        if self.model is None:
            return ValidationResult(
                name="Throughput",
                level=ValidationLevel.CRITICAL,
                status=ValidationStatus.SKIPPED,
                message="Model not loaded",
            )

        state = self._init_state()
        obs = jax.random.normal(jax.random.PRNGKey(0), (1, self.obs_dim))

        # Warmup
        for _ in range(5):
            self._run_inference(obs, state)

        # Measure throughput
        count = 0
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < 1.0:  # 1 second
            self._run_inference(obs, state)
            count += 1

        throughput_hz = count

        passed = throughput_hz >= self.TARGET_THROUGHPUT_HZ

        result = ValidationResult(
            name="Throughput",
            level=ValidationLevel.CRITICAL,
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            message=f"{throughput_hz:.1f} Hz (target: >={self.TARGET_THROUGHPUT_HZ} Hz)",
            metric_value=throughput_hz,
            metric_target=self.TARGET_THROUGHPUT_HZ,
            duration_ms=(time.time() - start) * 1000,
        )
        self._log(f"    {result.status.value.upper()}: {result.message}")
        return result

    def validate_output_stability(self) -> ValidationResult:
        """Validate outputs are stable and consistent."""
        self._log("\n[Test] Output Stability")
        start = time.time()

        if self.model is None:
            return ValidationResult(
                name="Output Stability",
                level=ValidationLevel.CRITICAL,
                status=ValidationStatus.SKIPPED,
                message="Model not loaded",
            )

        obs = jax.random.normal(jax.random.PRNGKey(42), (1, self.obs_dim))
        outputs = []

        for _ in range(10):
            state = self._init_state()
            out, _ = self._run_inference(obs, state)
            outputs.append(np.array(out[0]))

        # Check consistency
        diffs = [np.linalg.norm(outputs[0] - outputs[i]) for i in range(1, len(outputs))]
        max_diff = max(diffs)
        passed = max_diff < 1e-5

        result = ValidationResult(
            name="Output Stability",
            level=ValidationLevel.CRITICAL,
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            message=f"Max diff: {max_diff:.2e} (deterministic: {passed})",
            metric_value=max_diff,
            metric_target=1e-5,
            duration_ms=(time.time() - start) * 1000,
        )
        self._log(f"    {result.status.value.upper()}: {result.message}")
        return result

    def validate_nan_free(self) -> ValidationResult:
        """Validate no NaN values in outputs."""
        self._log("\n[Test] NaN-Free Outputs")
        start = time.time()

        if self.model is None:
            return ValidationResult(
                name="NaN-Free",
                level=ValidationLevel.CRITICAL,
                status=ValidationStatus.SKIPPED,
                message="Model not loaded",
            )

        nan_count = 0
        state = self._init_state()

        for i in range(100):
            obs = jax.random.normal(jax.random.PRNGKey(i), (1, self.obs_dim))
            out, state = self._run_inference(obs, state)

            if jnp.any(jnp.isnan(out)):
                nan_count += 1
            if jnp.any(jnp.isnan(state['s'])):
                nan_count += 1

        passed = nan_count == 0

        result = ValidationResult(
            name="NaN-Free",
            level=ValidationLevel.CRITICAL,
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            message=f"NaN occurrences: {nan_count} (target: 0)",
            metric_value=nan_count,
            metric_target=0,
            duration_ms=(time.time() - start) * 1000,
        )
        self._log(f"    {result.status.value.upper()}: {result.message}")
        return result

    def validate_state_bounds(self) -> ValidationResult:
        """Validate states stay within bounds under stress."""
        self._log("\n[Test] State Bounds")
        start = time.time()

        if self.model is None:
            return ValidationResult(
                name="State Bounds",
                level=ValidationLevel.CRITICAL,
                status=ValidationStatus.SKIPPED,
                message="Model not loaded",
            )

        max_state = 0.0
        state = self._init_state()

        # Stress test with large inputs
        for i in range(self.STRESS_TEST_ITERATIONS):
            obs = jax.random.normal(jax.random.PRNGKey(i), (1, self.obs_dim)) * 10
            _, state = self._run_inference(obs, state)
            max_state = max(max_state, float(jnp.max(jnp.abs(state['s']))))

        passed = max_state < self.MAX_STATE_VALUE

        result = ValidationResult(
            name="State Bounds",
            level=ValidationLevel.CRITICAL,
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            message=f"Max state: {max_state:.2f} (limit: {self.MAX_STATE_VALUE})",
            metric_value=max_state,
            metric_target=self.MAX_STATE_VALUE,
            duration_ms=(time.time() - start) * 1000,
        )
        self._log(f"    {result.status.value.upper()}: {result.message}")
        return result

    def validate_differentiation(self) -> ValidationResult:
        """Validate different inputs produce different outputs."""
        self._log("\n[Test] Input Differentiation")
        start = time.time()

        if self.model is None:
            return ValidationResult(
                name="Differentiation",
                level=ValidationLevel.IMPORTANT,
                status=ValidationStatus.SKIPPED,
                message="Model not loaded",
            )

        outputs = []
        for i in range(10):
            state = self._init_state()
            obs = jax.random.normal(jax.random.PRNGKey(i * 100), (1, self.obs_dim))
            out, _ = self._run_inference(obs, state)
            outputs.append(np.array(out[0]))

        # Calculate pairwise differences
        diffs = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                diffs.append(np.linalg.norm(outputs[i] - outputs[j]))

        avg_diff = np.mean(diffs)
        passed = avg_diff > 0.1

        result = ValidationResult(
            name="Differentiation",
            level=ValidationLevel.IMPORTANT,
            status=ValidationStatus.PASSED if passed else ValidationStatus.WARNING,
            message=f"Avg output diff: {avg_diff:.4f} (min expected: 0.1)",
            metric_value=avg_diff,
            metric_target=0.1,
            duration_ms=(time.time() - start) * 1000,
        )
        self._log(f"    {result.status.value.upper()}: {result.message}")
        return result

    def validate_memory_evolution(self) -> ValidationResult:
        """Validate CMS memory evolves over time."""
        self._log("\n[Test] Memory Evolution")
        start = time.time()

        if self.model is None:
            return ValidationResult(
                name="Memory Evolution",
                level=ValidationLevel.IMPORTANT,
                status=ValidationStatus.SKIPPED,
                message="Model not loaded",
            )

        state = self._init_state()
        initial_mem = sum(float(jnp.sum(jnp.abs(m))) for m in state['cms_memories'])

        for i in range(20):
            obs = jax.random.normal(jax.random.PRNGKey(i), (1, self.obs_dim))
            _, state = self._run_inference(obs, state)

        final_mem = sum(float(jnp.sum(jnp.abs(m))) for m in state['cms_memories'])
        mem_change = abs(final_mem - initial_mem)

        passed = mem_change > 0.01

        result = ValidationResult(
            name="Memory Evolution",
            level=ValidationLevel.IMPORTANT,
            status=ValidationStatus.PASSED if passed else ValidationStatus.WARNING,
            message=f"Memory change: {mem_change:.4f} (min expected: 0.01)",
            metric_value=mem_change,
            metric_target=0.01,
            duration_ms=(time.time() - start) * 1000,
        )
        self._log(f"    {result.status.value.upper()}: {result.message}")
        return result

    def validate_action_mapping(self) -> ValidationResult:
        """Validate action mapper integration."""
        self._log("\n[Test] Action Mapping")
        start = time.time()

        try:
            from .capafo_action_mapper import CapafoActionMapper, MockHardwareBackend

            mapper = CapafoActionMapper(hardware_backend=MockHardwareBackend())

            # Test with random output
            test_output = np.random.randn(32) * 0.5
            test_output[24] = -1.0  # Don't trigger estop

            commands = mapper.map_output(test_output)
            results = mapper.execute_commands(commands)

            success_rate = sum(results.values()) / len(results) if results else 0

            passed = success_rate > 0.8

            result = ValidationResult(
                name="Action Mapping",
                level=ValidationLevel.IMPORTANT,
                status=ValidationStatus.PASSED if passed else ValidationStatus.WARNING,
                message=f"Command success rate: {success_rate*100:.1f}%",
                metric_value=success_rate,
                metric_target=0.8,
                details={"commands_generated": len(commands)},
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            result = ValidationResult(
                name="Action Mapping",
                level=ValidationLevel.IMPORTANT,
                status=ValidationStatus.FAILED,
                message=f"Error: {e}",
                duration_ms=(time.time() - start) * 1000,
            )

        self._log(f"    {result.status.value.upper()}: {result.message}")
        return result

    def validate_chat_integration(self) -> ValidationResult:
        """Validate chat integration layer."""
        self._log("\n[Test] Chat Integration")
        start = time.time()

        try:
            from .chat_integration import ChatIntegrationLayer

            chat = ChatIntegrationLayer()

            # Test messages
            test_messages = ["hello", "move forward", "stop"]
            responses = []

            for msg in test_messages:
                response = chat.chat(msg)
                responses.append(response)

            avg_latency = np.mean([r.latency_ms for r in responses])
            all_have_intent = all(r.intent != "unknown" for r in responses)

            passed = avg_latency < 500 and all_have_intent

            result = ValidationResult(
                name="Chat Integration",
                level=ValidationLevel.IMPORTANT,
                status=ValidationStatus.PASSED if passed else ValidationStatus.WARNING,
                message=f"Avg latency: {avg_latency:.1f}ms, Intent recognition: {all_have_intent}",
                metric_value=avg_latency,
                metric_target=500,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            result = ValidationResult(
                name="Chat Integration",
                level=ValidationLevel.IMPORTANT,
                status=ValidationStatus.FAILED,
                message=f"Error: {e}",
                duration_ms=(time.time() - start) * 1000,
            )

        self._log(f"    {result.status.value.upper()}: {result.message}")
        return result

    def run_all_validations(self) -> ValidationReport:
        """Run all validation tests."""
        from datetime import datetime

        self._log("=" * 60)
        self._log("PRODUCTION VALIDATION SUITE")
        self._log("=" * 60)

        # Load model if not loaded
        if self.model is None and self.model_path:
            self.load_model()

        results = []

        # Run all tests
        results.append(self.validate_latency())
        results.append(self.validate_throughput())
        results.append(self.validate_output_stability())
        results.append(self.validate_nan_free())
        results.append(self.validate_state_bounds())
        results.append(self.validate_differentiation())
        results.append(self.validate_memory_evolution())
        results.append(self.validate_action_mapping())
        results.append(self.validate_chat_integration())

        # Determine overall status
        if all(r.status == ValidationStatus.PASSED for r in results if r.level == ValidationLevel.CRITICAL):
            if all(r.status in [ValidationStatus.PASSED, ValidationStatus.SKIPPED] for r in results):
                overall = ValidationStatus.PASSED
            else:
                overall = ValidationStatus.WARNING
        else:
            overall = ValidationStatus.FAILED

        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            model_path=str(self.model_path) if self.model_path else "none",
            results=results,
            overall_status=overall,
        )

        # Print summary
        self._log("\n" + "=" * 60)
        self._log("VALIDATION SUMMARY")
        self._log("=" * 60)
        self._log(f"Overall: {overall.value.upper()}")
        self._log(f"Critical tests passed: {report.critical_passed}")
        summary = report.summary
        self._log(f"Results: {summary['passed']}/{summary['total']} passed, "
                  f"{summary['failed']} failed, {summary['warning']} warnings")

        return report

    def export_report(self, report: ValidationReport, output_path: Path) -> None:
        """Export validation report to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def convert_numpy(obj):
            """Convert numpy/JAX types to Python native types for JSON serialization."""
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif hasattr(obj, 'item'):  # numpy/JAX scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy/JAX array
                return obj.tolist()
            return obj

        with open(output_path, 'w') as f:
            json.dump(convert_numpy(report.to_dict()), f, indent=2)

        self._log(f"\nReport exported to: {output_path}")


if __name__ == "__main__":
    from pathlib import Path

    model_path = Path(__file__).parent.parent.parent / "models" / "e2e_benchmark"

    validator = ProductionValidator(model_path=model_path)
    report = validator.run_all_validations()

    # Export report
    output_path = Path(__file__).parent.parent.parent / "validation_report.json"
    validator.export_report(report, output_path)
