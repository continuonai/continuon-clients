"""
Production Readiness Agent

Main orchestrator that prepares the seed model for production deployment.

Runs:
1. Latency optimization
2. Capafo action mapping setup
3. Chat integration setup
4. Production validation
5. Export production-ready artifacts
"""

import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .latency_optimizer import LatencyOptimizer
from .capafo_action_mapper import CapafoActionMapper, CapafoConfig, CAPAFO_ACTION_MAP
from .chat_integration import ChatIntegrationLayer
from .production_validator import ProductionValidator, ValidationStatus


@dataclass
class ProductionArtifacts:
    """Collection of production-ready artifacts."""
    optimized_model_path: Optional[Path] = None
    lite_model_path: Optional[Path] = None
    action_mapping_path: Optional[Path] = None
    validation_report_path: Optional[Path] = None
    manifest_path: Optional[Path] = None
    ready_for_production: bool = False


@dataclass
class AgentConfig:
    """Configuration for the production agent."""
    # Input
    model_dir: Path = Path("models/e2e_benchmark")

    # Output
    output_dir: Path = Path("models/production")

    # Optimization
    optimize_latency: bool = True
    target_latency_ms: float = 100.0
    create_lite_model: bool = True

    # Validation
    run_validation: bool = True
    require_critical_pass: bool = True

    # Export
    export_action_mapping: bool = True
    export_chat_config: bool = True

    # Robot config
    robot_name: str = "Capafo"
    robot_version: str = "1.0"


class ProductionReadinessAgent:
    """
    Agent that prepares seed model for production deployment.

    Usage:
        agent = ProductionReadinessAgent(config)
        artifacts = agent.run()

        if artifacts.ready_for_production:
            print("Model is ready for deployment!")
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.artifacts = ProductionArtifacts()

        # Components
        self.optimizer = None
        self.mapper = None
        self.chat = None
        self.validator = None

        # State
        self.steps_completed: List[str] = []
        self.errors: List[str] = []
        self.start_time = None

    def _log(self, message: str, level: str = "INFO") -> None:
        """Log a message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def _step(self, name: str) -> None:
        """Mark a step as starting."""
        self._log(f"Starting: {name}")
        self.steps_completed.append(name)

    def run(self) -> ProductionArtifacts:
        """
        Run the full production readiness pipeline.

        Returns:
            ProductionArtifacts with paths to all generated files
        """
        self.start_time = time.time()

        print("=" * 70)
        print("PRODUCTION READINESS AGENT")
        print(f"Model: {self.config.model_dir}")
        print(f"Output: {self.config.output_dir}")
        print("=" * 70)

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Latency Optimization
            if self.config.optimize_latency:
                self._run_latency_optimization()

            # Step 2: Action Mapping Setup
            if self.config.export_action_mapping:
                self._run_action_mapping_setup()

            # Step 3: Chat Integration Setup
            if self.config.export_chat_config:
                self._run_chat_integration_setup()

            # Step 4: Production Validation
            if self.config.run_validation:
                self._run_validation()

            # Step 5: Export Manifest
            self._export_manifest()

            # Determine readiness
            self._determine_readiness()

        except Exception as e:
            self.errors.append(f"Pipeline error: {e}")
            self._log(f"Pipeline failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()

        # Final summary
        self._print_summary()

        return self.artifacts

    def _run_latency_optimization(self) -> None:
        """Run latency optimization."""
        self._step("Latency Optimization")

        self.optimizer = LatencyOptimizer(
            model_dir=self.config.model_dir,
        )

        # Find and load model
        params_path = self.config.model_dir / "params_step_16.pkl"
        if params_path.exists():
            self.optimizer.load_model(params_path)
        else:
            self.optimizer.load_model()

        # Run optimizations
        results = self.optimizer.run_all_optimizations()

        # Export optimized models
        export_paths = self.optimizer.export_optimized_model(
            self.config.output_dir / "optimized"
        )

        self.artifacts.optimized_model_path = export_paths.get('optimized')
        self.artifacts.lite_model_path = export_paths.get('lite')

        # Check if target met
        if not results.get("target_met", False):
            self._log(f"Latency target not met (target: {self.config.target_latency_ms}ms)", "WARNING")
            if results.get("lite_model_available"):
                self._log("Lite model available with better latency", "INFO")

    def _run_action_mapping_setup(self) -> None:
        """Setup Capafo action mapping."""
        self._step("Action Mapping Setup")

        capafo_config = CapafoConfig(
            name=self.config.robot_name,
            version=self.config.robot_version,
        )

        self.mapper = CapafoActionMapper(config=capafo_config)

        # Export mapping
        mapping_path = self.config.output_dir / "capafo_action_mapping.json"
        self.mapper.export_mapping(mapping_path)
        self.artifacts.action_mapping_path = mapping_path

        # Print action space info
        info = self.mapper.get_action_space_info()
        self._log(f"Action space: {info['total_channels']} channels")
        for action_type, channels in info['channels_by_type'].items():
            self._log(f"  {action_type}: {len(channels)} channels")

    def _run_chat_integration_setup(self) -> None:
        """Setup chat integration layer."""
        self._step("Chat Integration Setup")

        # Create chat layer config
        chat_config = {
            "encoder_model": "all-MiniLM-L6-v2",
            "obs_dim": 128,
            "action_dim": 32,
            "output_dim": 32,
            "max_history": 20,
            "model_path": str(self.artifacts.optimized_model_path or self.config.model_dir),
            "intent_templates": list(self._get_intent_templates().keys()),
        }

        # Export chat config
        chat_config_path = self.config.output_dir / "chat_config.json"
        with open(chat_config_path, 'w') as f:
            json.dump(chat_config, f, indent=2)

        self._log(f"Chat config exported to: {chat_config_path}")

        # Test chat integration
        try:
            self.chat = ChatIntegrationLayer(
                model_path=self.artifacts.optimized_model_path or self.config.model_dir,
            )

            test_response = self.chat.chat("hello")
            self._log(f"Chat test response: {test_response.text}")
            self._log(f"Chat latency: {test_response.latency_ms:.2f}ms")

        except Exception as e:
            self._log(f"Chat integration test failed: {e}", "WARNING")

    def _get_intent_templates(self) -> Dict[str, Any]:
        """Get intent templates from chat integration."""
        from .chat_integration import INTENT_TEMPLATES
        return INTENT_TEMPLATES

    def _run_validation(self) -> None:
        """Run production validation suite."""
        self._step("Production Validation")

        model_path = self.artifacts.optimized_model_path or self.config.model_dir

        self.validator = ProductionValidator(model_path=model_path)
        report = self.validator.run_all_validations()

        # Export report
        report_path = self.config.output_dir / "validation_report.json"
        self.validator.export_report(report, report_path)
        self.artifacts.validation_report_path = report_path

        # Check critical tests
        if not report.critical_passed:
            self.errors.append("Critical validation tests failed")
            self._log("Critical validation tests FAILED", "ERROR")
        else:
            self._log("All critical validation tests PASSED", "INFO")

    def _export_manifest(self) -> None:
        """Export production manifest."""
        self._step("Export Manifest")

        manifest = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "source_model": str(self.config.model_dir),
            "robot": {
                "name": self.config.robot_name,
                "version": self.config.robot_version,
            },
            "artifacts": {
                "optimized_model": str(self.artifacts.optimized_model_path) if self.artifacts.optimized_model_path else None,
                "lite_model": str(self.artifacts.lite_model_path) if self.artifacts.lite_model_path else None,
                "action_mapping": str(self.artifacts.action_mapping_path) if self.artifacts.action_mapping_path else None,
                "validation_report": str(self.artifacts.validation_report_path) if self.artifacts.validation_report_path else None,
            },
            "pipeline": {
                "steps_completed": self.steps_completed,
                "errors": self.errors,
                "duration_seconds": time.time() - self.start_time if self.start_time else 0,
            },
            "recommendations": self._generate_recommendations(),
        }

        manifest_path = self.config.output_dir / "production_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        self.artifacts.manifest_path = manifest_path
        self._log(f"Manifest exported to: {manifest_path}")

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        if self.validator:
            # Check latency
            latency_result = next(
                (r for r in self.validator.results if r.name == "Latency"),
                None
            )
            if latency_result and latency_result.metric_value:
                if latency_result.metric_value > self.config.target_latency_ms:
                    recommendations.append(
                        f"Consider using lite model for real-time control "
                        f"(current: {latency_result.metric_value:.1f}ms)"
                    )

            # Check throughput
            throughput_result = next(
                (r for r in self.validator.results if r.name == "Throughput"),
                None
            )
            if throughput_result and throughput_result.metric_value:
                if throughput_result.metric_value < 20:
                    recommendations.append(
                        "Throughput below 20Hz - consider hardware acceleration"
                    )

        if not recommendations:
            recommendations.append("Model meets all production requirements")

        return recommendations

    def _determine_readiness(self) -> None:
        """Determine if model is ready for production."""
        ready = True

        # Check for errors
        if self.errors:
            ready = False

        # Check validation
        if self.config.require_critical_pass and self.validator:
            report = self.validator.run_all_validations()
            if not report.critical_passed:
                ready = False

        self.artifacts.ready_for_production = ready

    def _print_summary(self) -> None:
        """Print final summary."""
        duration = time.time() - self.start_time if self.start_time else 0

        print("\n" + "=" * 70)
        print("PRODUCTION READINESS SUMMARY")
        print("=" * 70)

        print(f"\nSteps completed: {len(self.steps_completed)}")
        for step in self.steps_completed:
            print(f"  [OK] {step}")

        if self.errors:
            print(f"\nErrors: {len(self.errors)}")
            for error in self.errors:
                print(f"  [ERROR] {error}")

        print(f"\nArtifacts:")
        print(f"  Optimized model: {self.artifacts.optimized_model_path or 'N/A'}")
        print(f"  Lite model: {self.artifacts.lite_model_path or 'N/A'}")
        print(f"  Action mapping: {self.artifacts.action_mapping_path or 'N/A'}")
        print(f"  Validation report: {self.artifacts.validation_report_path or 'N/A'}")
        print(f"  Manifest: {self.artifacts.manifest_path or 'N/A'}")

        print(f"\nDuration: {duration:.2f} seconds")

        status = "READY" if self.artifacts.ready_for_production else "NOT READY"
        print(f"\n{'='*70}")
        print(f"PRODUCTION STATUS: {status}")
        print(f"{'='*70}")


def run_production_agent(
    model_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> ProductionArtifacts:
    """
    Convenience function to run the production agent.

    Args:
        model_dir: Path to source model (default: models/e2e_benchmark)
        output_dir: Path for output (default: models/production)

    Returns:
        ProductionArtifacts with paths to all generated files
    """
    project_root = Path(__file__).parent.parent.parent

    config = AgentConfig(
        model_dir=model_dir or (project_root / "models" / "e2e_benchmark"),
        output_dir=output_dir or (project_root / "models" / "production"),
    )

    agent = ProductionReadinessAgent(config)
    return agent.run()


if __name__ == "__main__":
    artifacts = run_production_agent()

    if artifacts.ready_for_production:
        print("\nModel is ready for deployment!")
        print(f"Use artifacts from: {artifacts.manifest_path.parent if artifacts.manifest_path else 'N/A'}")
    else:
        print("\nModel needs additional work before deployment.")
