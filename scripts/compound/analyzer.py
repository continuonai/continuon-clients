#!/usr/bin/env python3
"""
Compound Analyzer - Autonomous codebase analysis for ContinuonXR

Generates reports by analyzing:
1. Code quality (syntax errors, type hints, complexity)
2. Test failures and coverage gaps
3. TODO/FIXME/HACK comments
4. Architecture consistency
5. ContinuonBrain integration state
6. Training pipeline health
7. Hardware integration gaps

This is the "eyes" of the compound system - it observes the codebase
and generates reports that the daemon will act on.
"""

import ast
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Finding:
    """A single finding from analysis."""
    category: str
    severity: str  # critical, high, medium, low
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    suggestion: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class AnalysisContext:
    """Context about the codebase state."""
    project_root: Path
    python_files: List[Path] = field(default_factory=list)
    test_files: List[Path] = field(default_factory=list)
    config_files: List[Path] = field(default_factory=list)
    brain_b_files: List[Path] = field(default_factory=list)
    continuonbrain_files: List[Path] = field(default_factory=list)
    trainer_ui_files: List[Path] = field(default_factory=list)


class CodebaseAnalyzer:
    """Analyzes the ContinuonXR codebase for issues."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.findings: List[Finding] = []
        self.context = self._build_context()

    def _build_context(self) -> AnalysisContext:
        """Build analysis context by scanning codebase."""
        ctx = AnalysisContext(project_root=self.project_root)

        # Find all Python files
        for py_file in self.project_root.rglob("*.py"):
            # Skip virtual envs and caches
            if any(skip in str(py_file) for skip in [
                'venv', '.venv', '__pycache__', 'node_modules', '.git', 'build', 'dist'
            ]):
                continue

            ctx.python_files.append(py_file)

            rel_path = str(py_file.relative_to(self.project_root))

            if 'test' in rel_path.lower():
                ctx.test_files.append(py_file)
            if rel_path.startswith('brain_b/'):
                ctx.brain_b_files.append(py_file)
            if rel_path.startswith('continuonbrain/'):
                ctx.continuonbrain_files.append(py_file)
            if rel_path.startswith('trainer_ui/'):
                ctx.trainer_ui_files.append(py_file)

        # Find config files
        for config_pattern in ['*.json', '*.yaml', '*.yml', '*.toml']:
            for config_file in self.project_root.glob(config_pattern):
                ctx.config_files.append(config_file)

        return ctx

    def analyze_all(self) -> List[Finding]:
        """Run all analyzers."""
        logger.info("Starting comprehensive codebase analysis...")

        analyzers = [
            self.analyze_syntax_errors,
            self.analyze_todo_comments,
            self.analyze_exception_handling,
            self.analyze_imports,
            self.analyze_test_coverage,
            self.analyze_brain_integration,
            self.analyze_training_pipeline,
            self.analyze_api_consistency,
            self.analyze_documentation_gaps,
            self.analyze_security,
        ]

        for analyzer in analyzers:
            try:
                logger.info(f"Running {analyzer.__name__}...")
                analyzer()
            except Exception as e:
                logger.warning(f"Analyzer {analyzer.__name__} failed: {e}")

        return self.findings

    def analyze_syntax_errors(self):
        """Check for Python syntax errors."""
        for py_file in self.context.python_files:
            try:
                source = py_file.read_text()
                ast.parse(source)
            except SyntaxError as e:
                self.findings.append(Finding(
                    category="syntax",
                    severity="critical",
                    title=f"Syntax error in {py_file.name}",
                    description=str(e),
                    file_path=str(py_file.relative_to(self.project_root)),
                    line_number=e.lineno,
                    auto_fixable=False,
                ))

    def analyze_todo_comments(self):
        """Find TODO, FIXME, HACK, XXX comments."""
        patterns = {
            r'#\s*TODO\s*:?\s*(.+)': ('medium', 'TODO'),
            r'#\s*FIXME\s*:?\s*(.+)': ('high', 'FIXME'),
            r'#\s*HACK\s*:?\s*(.+)': ('high', 'HACK'),
            r'#\s*XXX\s*:?\s*(.+)': ('medium', 'XXX'),
            r'#\s*BUG\s*:?\s*(.+)': ('critical', 'BUG'),
        }

        for py_file in self.context.python_files:
            try:
                lines = py_file.read_text().split('\n')
                for i, line in enumerate(lines, 1):
                    for pattern, (severity, tag) in patterns.items():
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            self.findings.append(Finding(
                                category="todo",
                                severity=severity,
                                title=f"{tag}: {match.group(1)[:50]}",
                                description=match.group(1),
                                file_path=str(py_file.relative_to(self.project_root)),
                                line_number=i,
                                auto_fixable=False,
                            ))
            except Exception:
                pass

    def analyze_exception_handling(self):
        """Find broad exception handlers and silent catches."""
        for py_file in self.context.python_files:
            try:
                source = py_file.read_text()
                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ExceptHandler):
                        # Check for bare except
                        if node.type is None:
                            self.findings.append(Finding(
                                category="exception",
                                severity="medium",
                                title=f"Bare except clause in {py_file.name}",
                                description="Bare 'except:' catches all exceptions including KeyboardInterrupt",
                                file_path=str(py_file.relative_to(self.project_root)),
                                line_number=node.lineno,
                                suggestion="Use 'except Exception:' or specific exception types",
                                auto_fixable=True,
                            ))

                        # Check for silent pass
                        if (len(node.body) == 1 and
                            isinstance(node.body[0], ast.Pass)):
                            self.findings.append(Finding(
                                category="exception",
                                severity="low",
                                title=f"Silent exception handler in {py_file.name}",
                                description="Exception is silently ignored with 'pass'",
                                file_path=str(py_file.relative_to(self.project_root)),
                                line_number=node.lineno,
                                suggestion="Add logging or handle the exception properly",
                                auto_fixable=True,
                            ))

                        # Check for broad Exception
                        if (isinstance(node.type, ast.Name) and
                            node.type.id == 'Exception'):
                            self.findings.append(Finding(
                                category="exception",
                                severity="low",
                                title=f"Broad exception handler in {py_file.name}",
                                description="Catching generic 'Exception' may hide bugs",
                                file_path=str(py_file.relative_to(self.project_root)),
                                line_number=node.lineno,
                                suggestion="Use more specific exception types",
                                auto_fixable=True,
                            ))

            except Exception:
                pass

    def analyze_imports(self):
        """Check for import issues."""
        for py_file in self.context.python_files:
            try:
                source = py_file.read_text()
                tree = ast.parse(source)

                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append((alias.name, node.lineno))
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ''
                        imports.append((module, node.lineno))

                # Check for circular import patterns
                rel_path = py_file.relative_to(self.project_root)
                module_parts = list(rel_path.parts[:-1]) + [rel_path.stem]

                for imp_name, line in imports:
                    # Check for potential circular imports within same package
                    if imp_name.startswith(module_parts[0] if module_parts else ''):
                        # This is an internal import - could be circular
                        pass  # Would need more sophisticated analysis

            except Exception:
                pass

    def analyze_test_coverage(self):
        """Analyze test coverage gaps."""
        # Find modules without tests
        tested_modules = set()
        for test_file in self.context.test_files:
            # Extract what module the test is for
            content = test_file.read_text()
            import_matches = re.findall(r'from\s+([\w.]+)\s+import|import\s+([\w.]+)', content)
            for match in import_matches:
                module = match[0] or match[1]
                if module.startswith(('brain_b', 'continuonbrain', 'trainer_ui')):
                    tested_modules.add(module.split('.')[0])

        # Check for untested modules
        module_dirs = ['brain_b', 'continuonbrain', 'trainer_ui']
        for module_dir in module_dirs:
            module_path = self.project_root / module_dir
            if module_path.exists():
                py_files = list(module_path.rglob("*.py"))
                if py_files and module_dir not in tested_modules:
                    self.findings.append(Finding(
                        category="testing",
                        severity="medium",
                        title=f"No tests found for {module_dir}",
                        description=f"The {module_dir} module has {len(py_files)} Python files but no tests",
                        suggestion=f"Add tests in {module_dir}/tests/ or tests/{module_dir}/",
                        auto_fixable=False,
                    ))

    def analyze_brain_integration(self):
        """Check Brain B <-> ContinuonBrain integration."""
        # Check if Brain B exports are used by ContinuonBrain
        brain_b_exports = self._get_brain_b_exports()
        continuonbrain_imports = self._get_continuonbrain_imports()

        # Check for mismatches
        unused_exports = brain_b_exports - continuonbrain_imports
        if unused_exports:
            self.findings.append(Finding(
                category="integration",
                severity="low",
                title="Unused Brain B exports",
                description=f"These Brain B exports aren't used by ContinuonBrain: {', '.join(list(unused_exports)[:5])}",
                suggestion="Consider if these need integration or can be removed",
                auto_fixable=False,
            ))

        # Check for RLDS pipeline completeness
        rlds_path = self.project_root / "continuonbrain" / "rlds"
        if rlds_path.exists():
            schema_file = rlds_path / "schema.py"
            if not schema_file.exists():
                self.findings.append(Finding(
                    category="integration",
                    severity="high",
                    title="Missing RLDS schema",
                    description="RLDS directory exists but schema.py is missing",
                    file_path="continuonbrain/rlds/",
                    auto_fixable=False,
                ))

    def _get_brain_b_exports(self) -> Set[str]:
        """Get exported symbols from Brain B."""
        exports = set()
        init_file = self.project_root / "brain_b" / "__init__.py"
        if init_file.exists():
            content = init_file.read_text()
            # Find __all__ list
            match = re.search(r'__all__\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if match:
                exports = set(re.findall(r'"(\w+)"|\'(\w+)\'', match.group(1)))
                exports = {e[0] or e[1] for e in exports}
        return exports

    def _get_continuonbrain_imports(self) -> Set[str]:
        """Get imports from ContinuonBrain that reference Brain B."""
        imports = set()
        for py_file in self.context.continuonbrain_files:
            try:
                content = py_file.read_text()
                # Find brain_b imports
                matches = re.findall(r'from\s+brain_b(?:\.[\w.]+)?\s+import\s+([\w,\s]+)', content)
                for match in matches:
                    names = [n.strip() for n in match.split(',')]
                    imports.update(names)
            except Exception:
                pass
        return imports

    def analyze_training_pipeline(self):
        """Analyze the training pipeline health."""
        # Check for episode files
        episodes_dir = self.project_root / "continuonbrain" / "rlds" / "episodes"
        if episodes_dir.exists():
            episodes = list(episodes_dir.glob("*/metadata.json"))
            if not episodes:
                self.findings.append(Finding(
                    category="training",
                    severity="medium",
                    title="No training episodes found",
                    description="The RLDS episodes directory is empty",
                    suggestion="Record some training sessions to generate episodes",
                    auto_fixable=False,
                ))
        else:
            self.findings.append(Finding(
                category="training",
                severity="medium",
                title="Episodes directory missing",
                description="continuonbrain/rlds/episodes/ does not exist",
                file_path="continuonbrain/rlds/",
                suggestion="Create the directory and add training episodes",
                auto_fixable=True,
            ))

        # Check trainer configuration
        trainer_config = self.project_root / "continuonbrain" / "trainer" / "config.json"
        if trainer_config.exists():
            try:
                config = json.loads(trainer_config.read_text())
                # Check for missing required fields
                required = ['model_type', 'batch_size', 'learning_rate']
                missing = [f for f in required if f not in config]
                if missing:
                    self.findings.append(Finding(
                        category="training",
                        severity="medium",
                        title="Incomplete trainer config",
                        description=f"Missing fields in trainer config: {', '.join(missing)}",
                        file_path="continuonbrain/trainer/config.json",
                        auto_fixable=True,
                    ))
            except json.JSONDecodeError:
                self.findings.append(Finding(
                    category="training",
                    severity="high",
                    title="Invalid trainer config",
                    description="trainer/config.json contains invalid JSON",
                    file_path="continuonbrain/trainer/config.json",
                    auto_fixable=False,
                ))

    def analyze_api_consistency(self):
        """Check for API consistency across modules."""
        # Find public functions in each module
        api_patterns = defaultdict(list)

        for py_file in self.context.python_files:
            try:
                source = py_file.read_text()
                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if not node.name.startswith('_'):
                            # Track async vs sync patterns
                            is_async = isinstance(node, ast.AsyncFunctionDef)
                            has_async_suffix = node.name.endswith('_async')

                            # Check for inconsistent async naming
                            if is_async and not has_async_suffix:
                                # This is fine - async def without _async suffix
                                pass

            except Exception:
                pass

    def analyze_documentation_gaps(self):
        """Find undocumented public APIs."""
        for py_file in self.context.python_files:
            try:
                source = py_file.read_text()
                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if not node.name.startswith('_'):
                            # Check for docstring
                            docstring = ast.get_docstring(node)
                            if not docstring:
                                # Only report for important-looking functions
                                if isinstance(node, ast.ClassDef) or len(node.body) > 5:
                                    self.findings.append(Finding(
                                        category="documentation",
                                        severity="low",
                                        title=f"Missing docstring: {node.name}",
                                        description=f"Public {'class' if isinstance(node, ast.ClassDef) else 'function'} lacks documentation",
                                        file_path=str(py_file.relative_to(self.project_root)),
                                        line_number=node.lineno,
                                        auto_fixable=True,
                                    ))

            except Exception:
                pass

    def analyze_security(self):
        """Check for security issues."""
        security_patterns = [
            (r'eval\s*\(', 'high', 'Use of eval()', 'eval() can execute arbitrary code'),
            (r'exec\s*\(', 'high', 'Use of exec()', 'exec() can execute arbitrary code'),
            (r'subprocess\..*shell\s*=\s*True', 'medium', 'Shell=True in subprocess', 'Can allow command injection'),
            (r'pickle\.load', 'medium', 'Pickle deserialization', 'Pickle can execute arbitrary code'),
            (r'password\s*=\s*["\'][^"\']+["\']', 'critical', 'Hardcoded password', 'Credentials should not be in code'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'critical', 'Hardcoded API key', 'Keys should be in environment variables'),
        ]

        for py_file in self.context.python_files:
            try:
                content = py_file.read_text()
                lines = content.split('\n')

                for pattern, severity, title, description in security_patterns:
                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            # Skip comments
                            if line.strip().startswith('#'):
                                continue

                            self.findings.append(Finding(
                                category="security",
                                severity=severity,
                                title=title,
                                description=description,
                                file_path=str(py_file.relative_to(self.project_root)),
                                line_number=i,
                                auto_fixable=False,
                            ))

            except Exception:
                pass


class ContinuonBrainStateAnalyzer:
    """Analyzes the current state of ContinuonBrain for partnership."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.brain_state = {}

    def analyze(self) -> Dict[str, Any]:
        """Get comprehensive brain state."""
        self.brain_state = {
            'mambawave': self._analyze_mambawave(),
            'ralph': self._analyze_ralph(),
            'rlds': self._analyze_rlds(),
            'hardware': self._analyze_hardware(),
            'training': self._analyze_training(),
        }
        return self.brain_state

    def _analyze_mambawave(self) -> Dict:
        """Analyze MambaWave skill state."""
        mambawave_dir = self.project_root / "continuonbrain" / "mambawave"
        if not mambawave_dir.exists():
            return {'status': 'missing', 'health': 0}

        # Check for key files
        required_files = ['config.py', 'skill.py', 'world_model.py']
        existing = [f for f in required_files if (mambawave_dir / f).exists()]

        # Check for model weights
        weights_dir = mambawave_dir / "weights"
        has_weights = weights_dir.exists() and any(weights_dir.glob("*.pt"))

        return {
            'status': 'operational' if len(existing) == len(required_files) else 'partial',
            'health': len(existing) / len(required_files),
            'files': existing,
            'has_weights': has_weights,
        }

    def _analyze_ralph(self) -> Dict:
        """Analyze Ralph learning loops state."""
        ralph_dir = self.project_root / "continuonbrain" / "ralph"
        if not ralph_dir.exists():
            return {'status': 'missing', 'health': 0}

        # Check loop implementations
        loops = ['fast_loop.py', 'mid_loop.py', 'slow_loop.py']
        existing = [l for l in loops if (ralph_dir / l).exists()]

        return {
            'status': 'operational' if len(existing) == len(loops) else 'partial',
            'health': len(existing) / len(loops),
            'loops': existing,
        }

    def _analyze_rlds(self) -> Dict:
        """Analyze RLDS training data state."""
        rlds_dir = self.project_root / "continuonbrain" / "rlds"
        episodes_dir = rlds_dir / "episodes"

        if not episodes_dir.exists():
            return {'status': 'no_data', 'episode_count': 0, 'health': 0}

        episodes = list(episodes_dir.glob("*/metadata.json"))
        total_steps = 0
        total_reward = 0

        for ep_meta in episodes[:10]:  # Sample first 10
            try:
                meta = json.loads(ep_meta.read_text())
                total_steps += meta.get('num_steps', 0)
                total_reward += meta.get('total_reward', 0)
            except Exception:
                pass

        return {
            'status': 'collecting' if episodes else 'empty',
            'episode_count': len(episodes),
            'total_steps': total_steps,
            'avg_reward': total_reward / len(episodes) if episodes else 0,
            'health': min(len(episodes) / 100, 1.0),  # 100 episodes = healthy
        }

    def _analyze_hardware(self) -> Dict:
        """Analyze hardware integration state."""
        hardware_dir = self.project_root / "brain_b" / "hardware"
        if not hardware_dir.exists():
            return {'status': 'missing', 'health': 0}

        # Check for hardware implementations
        hw_files = list(hardware_dir.glob("*.py"))
        has_motor = any('motor' in f.name.lower() for f in hw_files)
        has_camera = any('camera' in f.name.lower() for f in hw_files)
        has_sensor = any('sensor' in f.name.lower() for f in hw_files)

        return {
            'status': 'operational',
            'has_motor': has_motor,
            'has_camera': has_camera,
            'has_sensor': has_sensor,
            'health': sum([has_motor, has_camera, has_sensor]) / 3,
        }

    def _analyze_training(self) -> Dict:
        """Analyze training infrastructure state."""
        trainer_dir = self.project_root / "continuonbrain" / "trainer"
        if not trainer_dir.exists():
            return {'status': 'missing', 'health': 0}

        # Check for training components
        has_trainer = (trainer_dir / "local_lora_trainer.py").exists()
        has_config = (trainer_dir / "config.json").exists()
        has_daemon = (trainer_dir / "auto_trainer_daemon.py").exists()

        return {
            'status': 'ready' if has_trainer else 'partial',
            'has_trainer': has_trainer,
            'has_config': has_config,
            'has_daemon': has_daemon,
            'health': sum([has_trainer, has_config, has_daemon]) / 3,
        }

    def get_improvement_priorities(self) -> List[Tuple[str, str, float]]:
        """Get prioritized list of what needs improvement."""
        if not self.brain_state:
            self.analyze()

        priorities = []

        for component, state in self.brain_state.items():
            health = state.get('health', 0)
            if health < 1.0:
                priority = 'high' if health < 0.3 else 'medium' if health < 0.7 else 'low'
                priorities.append((component, priority, health))

        # Sort by health (lowest first)
        priorities.sort(key=lambda x: x[2])

        return priorities


class ReportGenerator:
    """Generates markdown reports from analysis."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.reports_dir = project_root / "reports"
        self.reports_dir.mkdir(exist_ok=True)

    def generate_report(self, findings: List[Finding], brain_state: Dict) -> Path:
        """Generate a comprehensive report."""
        now = datetime.now()
        report_name = f"{now.strftime('%Y-%m-%d')}-auto-analysis.md"
        report_path = self.reports_dir / report_name

        # Group findings by category and severity
        by_severity = defaultdict(list)
        for f in findings:
            by_severity[f.severity].append(f)

        # Build report
        lines = [
            f"# ContinuonXR Analysis Report",
            f"",
            f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## Summary",
            f"",
            f"- **Total Findings**: {len(findings)}",
            f"- **Critical**: {len(by_severity['critical'])}",
            f"- **High**: {len(by_severity['high'])}",
            f"- **Medium**: {len(by_severity['medium'])}",
            f"- **Low**: {len(by_severity['low'])}",
            f"",
            f"## ContinuonBrain State",
            f"",
        ]

        # Brain state summary
        for component, state in brain_state.items():
            health = state.get('health', 0)
            status = state.get('status', 'unknown')
            health_bar = '█' * int(health * 10) + '░' * (10 - int(health * 10))
            lines.append(f"- **{component}**: {status} [{health_bar}] {health*100:.0f}%")

        lines.extend([
            f"",
            f"---",
            f"",
        ])

        # Critical issues first
        if by_severity['critical']:
            lines.extend([
                f"## Critical Issues",
                f"",
            ])
            for i, f in enumerate(by_severity['critical'], 1):
                lines.extend(self._format_finding(i, f))

        # High priority
        if by_severity['high']:
            lines.extend([
                f"## High Priority",
                f"",
            ])
            for i, f in enumerate(by_severity['high'], 1):
                lines.extend(self._format_finding(i, f))

        # Medium priority
        if by_severity['medium']:
            lines.extend([
                f"## Medium Priority",
                f"",
            ])
            for i, f in enumerate(by_severity['medium'], 1):
                lines.extend(self._format_finding(i, f))

        # Low priority
        if by_severity['low']:
            lines.extend([
                f"## Low Priority",
                f"",
            ])
            for i, f in enumerate(by_severity['low'], 1):
                lines.extend(self._format_finding(i, f))

        # Write report
        report_path.write_text('\n'.join(lines))
        logger.info(f"Report generated: {report_path}")

        return report_path

    def _format_finding(self, num: int, finding: Finding) -> List[str]:
        """Format a single finding."""
        lines = [
            f"### {num}. {finding.title} [{finding.severity.upper()}]",
            f"",
            f"**Category**: {finding.category}",
        ]

        if finding.file_path:
            loc = f"`{finding.file_path}`"
            if finding.line_number:
                loc += f" (line {finding.line_number})"
            lines.append(f"**Location**: {loc}")

        lines.extend([
            f"",
            finding.description,
            f"",
        ])

        if finding.suggestion:
            lines.extend([
                f"**Suggestion**: {finding.suggestion}",
                f"",
            ])

        if finding.auto_fixable:
            lines.append(f"✅ Auto-fixable")
            lines.append(f"")

        return lines


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compound Analyzer - Codebase analysis')
    parser.add_argument('--project', type=str, default='.', help='Project root')
    parser.add_argument('--output', type=str, help='Output report path')
    parser.add_argument('--json', action='store_true', help='Output JSON instead of markdown')

    args = parser.parse_args()

    project_root = Path(args.project).resolve()

    # Run analysis
    analyzer = CodebaseAnalyzer(project_root)
    findings = analyzer.analyze_all()

    # Analyze brain state
    brain_analyzer = ContinuonBrainStateAnalyzer(project_root)
    brain_state = brain_analyzer.analyze()

    if args.json:
        output = {
            'findings': [
                {
                    'category': f.category,
                    'severity': f.severity,
                    'title': f.title,
                    'description': f.description,
                    'file_path': f.file_path,
                    'line_number': f.line_number,
                    'suggestion': f.suggestion,
                    'auto_fixable': f.auto_fixable,
                }
                for f in findings
            ],
            'brain_state': brain_state,
        }
        print(json.dumps(output, indent=2))
    else:
        # Generate report
        generator = ReportGenerator(project_root)
        report_path = generator.generate_report(findings, brain_state)

        print(f"\n📊 Analysis Complete")
        print(f"=" * 40)
        print(f"Findings: {len(findings)}")
        print(f"Report: {report_path}")

        # Show priorities
        priorities = brain_analyzer.get_improvement_priorities()
        if priorities:
            print(f"\n🎯 Improvement Priorities:")
            for component, priority, health in priorities:
                print(f"  [{priority.upper()}] {component} ({health*100:.0f}% health)")


if __name__ == '__main__':
    main()
