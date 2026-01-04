"""
Benchmark Suite - Unified benchmark interface for autonomous training.

Wraps ProgressiveBenchmark and integrates with BenchmarkTracker for
historical tracking and automatic "beat previous best" progression.
"""

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from continuonbrain.services.benchmark_tracker import (
    BenchmarkTracker,
    BenchmarkResult,
    BenchmarkComparison,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRunResult:
    """Result of a full benchmark run."""
    run_id: str
    model_name: str
    model_version: str
    timestamp: float
    duration_seconds: float
    overall_score: float
    highest_level_passed: str
    total_tests: int
    tests_passed: int
    comparisons: List[BenchmarkComparison]
    new_bests_count: int
    level_summaries: Dict[str, Dict[str, Any]]


# Metric name mapping from test names to tracker metrics
METRIC_MAP = {
    # Level 1
    "Output Stability": "output_stability",
    "Non-trivial Output": "output_nontrivial",
    "Inference Latency": "inference_latency_ms",
    # Level 2
    "Command Differentiation": "command_differentiation",
    "State Evolution": "state_evolution",
    "Spatial Understanding": "spatial_understanding",
    # Level 3
    "Memory Persistence": "memory_persistence",
    "Context Switching": "context_switching",
    "Hierarchical Commands": "hierarchical_commands",
    # Level 4
    "Safety Priority": "safety_priority",
    "Error Recovery": "error_recovery",
    "Multi-step Planning": "multistep_planning",
    "Sensor Fusion Basic": "sensor_fusion_basic",
    "Sensor Degradation": "sensor_degradation",
    # Level 5
    "Self Monitoring": "self_monitoring",
    "Continuous Learning": "continuous_learning",
    "World Model Prediction": "world_model_prediction",
    "Sensor Fusion Multimodal": "sensor_fusion_multimodal",
    "Embodied Spatial Reasoning": "embodied_spatial_reasoning",
    # Level 6
    "Parts Inventory Understanding": "parts_inventory",
    "Build Plan Reasoning": "build_plan_reasoning",
    "Swarm Coordination": "swarm_coordination",
    "Clone Authorization": "clone_authorization",
    "Experience Sharing": "experience_sharing",
    "Lineage Awareness": "lineage_awareness",
}

# Category mapping
LEVEL_TO_CATEGORY = {
    "BASIC": "foundation",
    "INTERMEDIATE": "combined_skills",
    "ADVANCED": "complex_reasoning",
    "EXPERT": "real_world",
    "AUTONOMOUS": "self_directed",
    "SWARM": "multi_robot",
}


class BenchmarkSuite:
    """
    Unified benchmark suite for autonomous training.

    Integrates ProgressiveBenchmark with BenchmarkTracker for:
    - Running comprehensive evaluations
    - Recording results with history
    - Comparing to previous best scores
    - Tracking improvement over time
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        tracker: Optional[BenchmarkTracker] = None,
        use_lightweight_encoder: bool = False,
    ):
        """
        Initialize benchmark suite.

        Args:
            model_dir: Path to model directory
            tracker: BenchmarkTracker instance (creates one if not provided)
            use_lightweight_encoder: Use fast encoder for quick tests
        """
        self.model_dir = Path(model_dir or "/opt/continuonos/brain/model/seed_stable")
        self.tracker = tracker or BenchmarkTracker()
        self.use_lightweight_encoder = use_lightweight_encoder
        self._benchmark = None
        self._encoder = None

    def _ensure_loaded(self) -> None:
        """Lazy load the benchmark and encoder."""
        if self._benchmark is not None:
            return

        import os
        from continuonbrain.eval.encoder_cache import (
            get_cached_encoder,
            get_lightweight_encoder,
        )
        from continuonbrain.eval.progressive_benchmark import ProgressiveBenchmark

        logger.info("Loading encoder for benchmark suite...")
        if self.use_lightweight_encoder:
            self._encoder = get_lightweight_encoder(obs_dim=128)
            logger.info("Using lightweight encoder")
        else:
            self._encoder = get_cached_encoder(
                model_name='google/embeddinggemma-300m',
                token=os.environ.get('HUGGINGFACE_TOKEN'),
            )

        logger.info(f"Loading model from {self.model_dir}")
        self._benchmark = ProgressiveBenchmark(self.model_dir, self._encoder)
        logger.info(
            f"Loaded model v{self._benchmark.manifest['version']} "
            f"({self._benchmark.manifest['model']['param_count']:,} params)"
        )

    def run_full_benchmark(self) -> BenchmarkRunResult:
        """
        Run all benchmark tests and record results.

        Returns:
            BenchmarkRunResult with all comparisons and summaries
        """
        self._ensure_loaded()

        run_id = str(uuid.uuid4())[:8]
        model_version = self._benchmark.manifest['version']
        model_name = self._benchmark.manifest.get('model', {}).get('name', 'seed_model')

        logger.info(f"Starting benchmark run {run_id} for {model_name} v{model_version}")
        start_time = time.time()

        # Run all progressive tests
        prog_result = self._benchmark.run_all()

        duration = time.time() - start_time
        timestamp = time.time()

        # Record each test result and collect comparisons
        comparisons = []
        for test_result in prog_result.results:
            metric_name = METRIC_MAP.get(test_result.name, test_result.name.lower().replace(" ", "_"))
            category = LEVEL_TO_CATEGORY.get(test_result.level.name, "other")

            benchmark_result = BenchmarkResult(
                benchmark_id=run_id,
                category=category,
                metric_name=metric_name,
                value=test_result.score,
                model_name=model_name,
                model_version=model_version,
                timestamp=timestamp,
                metadata={
                    "test_name": test_result.name,
                    "level": test_result.level.name,
                    "passed": test_result.passed,
                    **test_result.details,
                },
            )

            comparison = self.tracker.record(benchmark_result)
            comparisons.append(comparison)

        # Also record aggregate metrics
        aggregate_metrics = [
            ("overall_score", prog_result.overall_score, "aggregate"),
            ("highest_level", float(prog_result.highest_level_passed.value), "aggregate"),
        ]

        for metric_name, value, category in aggregate_metrics:
            benchmark_result = BenchmarkResult(
                benchmark_id=run_id,
                category=category,
                metric_name=metric_name,
                value=value,
                model_name=model_name,
                model_version=model_version,
                timestamp=timestamp,
            )
            comparison = self.tracker.record(benchmark_result)
            comparisons.append(comparison)

        # Build level summaries
        level_summaries = {}
        for level in prog_result.results[0].level.__class__:
            summary = prog_result.get_level_summary(level)
            level_summaries[level.name] = {
                "passed": summary.passed,
                "tests_passed": summary.tests_passed,
                "tests_total": summary.tests_total,
                "avg_score": summary.avg_score,
            }

        new_bests = sum(1 for c in comparisons if c.is_new_best)

        logger.info(
            f"Benchmark run {run_id} complete: "
            f"score={prog_result.overall_score:.3f}, "
            f"level={prog_result.highest_level_passed.name}, "
            f"new_bests={new_bests}"
        )

        return BenchmarkRunResult(
            run_id=run_id,
            model_name=model_name,
            model_version=model_version,
            timestamp=timestamp,
            duration_seconds=duration,
            overall_score=prog_result.overall_score,
            highest_level_passed=prog_result.highest_level_passed.name,
            total_tests=len(prog_result.results),
            tests_passed=sum(1 for r in prog_result.results if r.passed),
            comparisons=comparisons,
            new_bests_count=new_bests,
            level_summaries=level_summaries,
        )

    def run_quick_benchmark(self) -> BenchmarkRunResult:
        """
        Run a quick subset of benchmarks for faster feedback.

        Runs one test per level for a quick health check.
        """
        self._ensure_loaded()

        run_id = str(uuid.uuid4())[:8]
        model_version = self._benchmark.manifest['version']
        model_name = self._benchmark.manifest.get('model', {}).get('name', 'seed_model')

        logger.info(f"Starting quick benchmark {run_id}")
        start_time = time.time()
        timestamp = time.time()

        # Run representative tests from each level
        quick_tests = [
            ("L1", self._benchmark.test_L1_inference_latency),
            ("L2", self._benchmark.test_L2_command_differentiation),
            ("L3", self._benchmark.test_L3_memory_persistence),
            ("L4", self._benchmark.test_L4_safety_priority),
            ("L5", self._benchmark.test_L5_continuous_learning),
            ("L6", self._benchmark.test_L6_swarm_coordination),
        ]

        comparisons = []
        results = []

        for level_name, test_fn in quick_tests:
            try:
                test_result = test_fn()
                results.append(test_result)

                metric_name = METRIC_MAP.get(
                    test_result.name, test_result.name.lower().replace(" ", "_")
                )
                category = LEVEL_TO_CATEGORY.get(test_result.level.name, "other")

                benchmark_result = BenchmarkResult(
                    benchmark_id=run_id,
                    category=category,
                    metric_name=metric_name,
                    value=test_result.score,
                    model_name=model_name,
                    model_version=model_version,
                    timestamp=timestamp,
                    metadata={
                        "test_name": test_result.name,
                        "level": test_result.level.name,
                        "passed": test_result.passed,
                        "quick_run": True,
                    },
                )

                comparison = self.tracker.record(benchmark_result)
                comparisons.append(comparison)

            except Exception as e:
                logger.warning(f"Quick test {level_name} failed: {e}")

        duration = time.time() - start_time
        overall_score = sum(r.score for r in results) / len(results) if results else 0.0
        tests_passed = sum(1 for r in results if r.passed)

        # Determine highest level passed
        level_names = ["BASIC", "INTERMEDIATE", "ADVANCED", "EXPERT", "AUTONOMOUS", "SWARM"]
        highest = "BASIC"
        for i, (level_name, test_fn) in enumerate(quick_tests):
            if i < len(results) and results[i].passed:
                highest = results[i].level.name
            else:
                break

        new_bests = sum(1 for c in comparisons if c.is_new_best)

        logger.info(f"Quick benchmark complete: score={overall_score:.3f}, new_bests={new_bests}")

        return BenchmarkRunResult(
            run_id=run_id,
            model_name=model_name,
            model_version=model_version,
            timestamp=timestamp,
            duration_seconds=duration,
            overall_score=overall_score,
            highest_level_passed=highest,
            total_tests=len(results),
            tests_passed=tests_passed,
            comparisons=comparisons,
            new_bests_count=new_bests,
            level_summaries={},  # Not computed for quick run
        )

    def run_level_benchmark(self, level: int) -> List[BenchmarkComparison]:
        """
        Run benchmarks for a specific level only.

        Args:
            level: Level number (1-6)

        Returns:
            List of comparisons for that level's tests
        """
        self._ensure_loaded()

        level_tests = {
            1: [
                self._benchmark.test_L1_output_stability,
                self._benchmark.test_L1_output_nonzero,
                self._benchmark.test_L1_inference_latency,
            ],
            2: [
                self._benchmark.test_L2_command_differentiation,
                self._benchmark.test_L2_state_evolution,
                self._benchmark.test_L2_spatial_understanding,
            ],
            3: [
                self._benchmark.test_L3_memory_persistence,
                self._benchmark.test_L3_context_switching,
                self._benchmark.test_L3_hierarchical_commands,
            ],
            4: [
                self._benchmark.test_L4_safety_priority,
                self._benchmark.test_L4_error_recovery,
                self._benchmark.test_L4_multi_step_planning,
                self._benchmark.test_L4_sensor_fusion_basic,
                self._benchmark.test_L4_sensor_degradation,
            ],
            5: [
                self._benchmark.test_L5_self_monitoring,
                self._benchmark.test_L5_continuous_learning,
                self._benchmark.test_L5_world_model,
                self._benchmark.test_L5_sensor_fusion_multimodal,
                self._benchmark.test_L5_embodied_spatial_reasoning,
            ],
            6: [
                self._benchmark.test_L6_parts_inventory_understanding,
                self._benchmark.test_L6_build_plan_reasoning,
                self._benchmark.test_L6_swarm_coordination,
                self._benchmark.test_L6_clone_authorization,
                self._benchmark.test_L6_experience_sharing,
                self._benchmark.test_L6_lineage_awareness,
            ],
        }

        if level not in level_tests:
            raise ValueError(f"Invalid level {level}. Must be 1-6.")

        run_id = str(uuid.uuid4())[:8]
        model_version = self._benchmark.manifest['version']
        model_name = self._benchmark.manifest.get('model', {}).get('name', 'seed_model')
        timestamp = time.time()

        logger.info(f"Running level {level} benchmark")

        comparisons = []
        for test_fn in level_tests[level]:
            try:
                test_result = test_fn()
                metric_name = METRIC_MAP.get(
                    test_result.name, test_result.name.lower().replace(" ", "_")
                )
                category = LEVEL_TO_CATEGORY.get(test_result.level.name, "other")

                benchmark_result = BenchmarkResult(
                    benchmark_id=run_id,
                    category=category,
                    metric_name=metric_name,
                    value=test_result.score,
                    model_name=model_name,
                    model_version=model_version,
                    timestamp=timestamp,
                    metadata={
                        "test_name": test_result.name,
                        "level": test_result.level.name,
                        "passed": test_result.passed,
                    },
                )

                comparison = self.tracker.record(benchmark_result)
                comparisons.append(comparison)

            except Exception as e:
                logger.warning(f"Test {test_fn.__name__} failed: {e}")

        return comparisons

    def get_improvement_report(self) -> Dict[str, Any]:
        """
        Get a report of overall improvement trends.

        Returns:
            Dict with trend analysis for key metrics
        """
        key_metrics = [
            "overall_score",
            "inference_latency_ms",
            "memory_persistence",
            "safety_priority",
            "continuous_learning",
        ]

        report = {
            "metrics": {},
            "summary": self.tracker.get_summary(),
        }

        for metric in key_metrics:
            trend = self.tracker.get_trend(metric, window=10)
            best = self.tracker.get_best(metric)
            report["metrics"][metric] = {
                "trend": trend,
                "best": best,
            }

        return report

    def should_trigger_training(self, threshold: float = 0.01) -> Tuple[bool, str]:
        """
        Determine if training should be triggered based on benchmark results.

        Args:
            threshold: Minimum improvement threshold to consider progress

        Returns:
            Tuple of (should_train, reason)
        """
        summary = self.tracker.get_summary()

        if summary["total_benchmarks"] < 5:
            return True, "Not enough benchmark history"

        # Check trends for key metrics
        declining_metrics = []
        for metric in ["overall_score", "memory_persistence", "safety_priority"]:
            trend = self.tracker.get_trend(metric)
            if trend.get("trend") == "degrading":
                declining_metrics.append(metric)

        if declining_metrics:
            return True, f"Declining metrics: {', '.join(declining_metrics)}"

        # Check if we've plateaued
        overall_trend = self.tracker.get_trend("overall_score", window=5)
        if overall_trend.get("trend") == "stable":
            # Stable but not at max potential
            latest = overall_trend.get("latest_value", 0)
            if latest < 0.8:
                return True, f"Plateaued at score {latest:.2f}, room for improvement"

        return False, "Performance is stable or improving"


def run_benchmark_suite(
    model_dir: str = "/opt/continuonos/brain/model/seed_stable",
    quick: bool = False,
    output_file: Optional[str] = None,
) -> BenchmarkRunResult:
    """
    Convenience function to run benchmarks.

    Args:
        model_dir: Path to model directory
        quick: Run quick benchmark instead of full
        output_file: Optional file to save results

    Returns:
        BenchmarkRunResult
    """
    import json

    suite = BenchmarkSuite(model_dir=Path(model_dir))

    if quick:
        result = suite.run_quick_benchmark()
    else:
        result = suite.run_full_benchmark()

    if output_file:
        output_data = {
            "run_id": result.run_id,
            "model_name": result.model_name,
            "model_version": result.model_version,
            "timestamp": result.timestamp,
            "duration_seconds": result.duration_seconds,
            "overall_score": result.overall_score,
            "highest_level_passed": result.highest_level_passed,
            "total_tests": result.total_tests,
            "tests_passed": result.tests_passed,
            "new_bests_count": result.new_bests_count,
            "level_summaries": result.level_summaries,
            "comparisons": [
                {
                    "metric_name": c.metric_name,
                    "current_value": c.current_value,
                    "best_value": c.best_value,
                    "improvement": c.improvement,
                    "is_new_best": c.is_new_best,
                }
                for c in result.comparisons
            ],
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    return result


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run benchmark suite")
    parser.add_argument("--model-dir", default="/opt/continuonos/brain/model/seed_stable")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--output", default=None, help="Output file for results")
    args = parser.parse_args()

    result = run_benchmark_suite(
        model_dir=args.model_dir,
        quick=args.quick,
        output_file=args.output,
    )

    print(f"\n{'='*60}")
    print(f"Benchmark Run: {result.run_id}")
    print(f"{'='*60}")
    print(f"Model: {result.model_name} v{result.model_version}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"Highest Level: {result.highest_level_passed}")
    print(f"Tests Passed: {result.tests_passed}/{result.total_tests}")
    print(f"New Best Scores: {result.new_bests_count}")
    print(f"{'='*60}")

    # Exit with non-zero if score is too low
    sys.exit(0 if result.overall_score >= 0.3 else 1)
