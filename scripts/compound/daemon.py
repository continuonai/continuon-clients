#!/usr/bin/env python3
"""
Compound Daemon - Autonomous self-improvement system for ContinuonXR

A self-aware system that understands ContinuonBrain's state and works
as a partner to continuously improve the codebase.

Runs continuously in the background:
1. Analyzes codebase and generates reports (self-aware)
2. Understands ContinuonBrain health and priorities
3. Queues and prioritizes fixes
4. Spawns Claude Code to implement each fix
5. Runs quality checks
6. Auto-creates PRs and merges on success
7. Learns from outcomes to improve future fixes

Usage:
    python scripts/compound/daemon.py              # Run daemon
    python scripts/compound/daemon.py --once       # Single cycle
    python scripts/compound/daemon.py --status     # Check status
    python scripts/compound/daemon.py --analyze    # Just analyze
"""

import json
import os
import re
import subprocess
import sys
import time
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum

# Import our analyzer
try:
    from analyzer import (
        CodebaseAnalyzer,
        ContinuonBrainStateAnalyzer,
        ReportGenerator,
        Finding,
    )
except ImportError:
    # Running from different directory
    sys.path.insert(0, str(Path(__file__).parent))
    from analyzer import (
        CodebaseAnalyzer,
        ContinuonBrainStateAnalyzer,
        ReportGenerator,
        Finding,
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('compound_daemon.log')
    ]
)
logger = logging.getLogger(__name__)


class IssueStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class IssuePriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Issue:
    """A single issue extracted from a report."""
    id: str
    title: str
    description: str
    priority: IssuePriority
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    source_report: str = ""
    status: IssueStatus = IssueStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    last_error: Optional[str] = None
    pr_number: Optional[int] = None
    branch_name: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Issue':
        data['priority'] = IssuePriority(data['priority'])
        data['status'] = IssueStatus(data['status'])
        return cls(**data)


@dataclass
class DaemonState:
    """Persistent state for the compound daemon."""
    issues: List[Issue] = field(default_factory=list)
    processed_reports: Dict[str, str] = field(default_factory=dict)  # path -> hash
    current_issue_id: Optional[str] = None
    total_fixes: int = 0
    total_failures: int = 0
    last_run: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            'issues': [i.to_dict() for i in self.issues],
            'processed_reports': self.processed_reports,
            'current_issue_id': self.current_issue_id,
            'total_fixes': self.total_fixes,
            'total_failures': self.total_failures,
            'last_run': self.last_run,
            'started_at': self.started_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DaemonState':
        issues = [Issue.from_dict(i) for i in data.get('issues', [])]
        return cls(
            issues=issues,
            processed_reports=data.get('processed_reports', {}),
            current_issue_id=data.get('current_issue_id'),
            total_fixes=data.get('total_fixes', 0),
            total_failures=data.get('total_failures', 0),
            last_run=data.get('last_run'),
            started_at=data.get('started_at', datetime.now().isoformat()),
        )


class CompoundDaemon:
    """Autonomous compound-product daemon."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.reports_dir = project_root / "reports"
        self.state_file = project_root / "scripts" / "compound" / "state.json"
        self.config_file = project_root / "compound.config.json"
        self.state = self._load_state()
        self.config = self._load_config()

        # Ensure directories exist
        self.reports_dir.mkdir(exist_ok=True)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> DaemonState:
        """Load or create daemon state."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return DaemonState.from_dict(json.load(f))
            except Exception as e:
                logger.warning(f"Could not load state: {e}, starting fresh")
        return DaemonState()

    def _save_state(self):
        """Persist daemon state."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def _load_config(self) -> Dict:
        """Load compound config."""
        if self.config_file.exists():
            with open(self.config_file) as f:
                return json.load(f)
        return {
            "qualityChecks": [
                "python -m py_compile brain_b/main.py",
                "python -m py_compile trainer_ui/server.py",
            ],
            "branchPrefix": "compound/",
            "maxIterations": 25,
        }

    def _hash_file(self, path: Path) -> str:
        """Get hash of file contents."""
        return hashlib.md5(path.read_bytes()).hexdigest()

    def scan_reports(self) -> List[Issue]:
        """Scan reports directory for new/changed reports."""
        new_issues = []

        for report_path in self.reports_dir.glob("*.md"):
            file_hash = self._hash_file(report_path)
            str_path = str(report_path)

            # Skip if already processed and unchanged
            if str_path in self.state.processed_reports:
                if self.state.processed_reports[str_path] == file_hash:
                    continue

            logger.info(f"Processing report: {report_path.name}")
            issues = self._parse_report(report_path)
            new_issues.extend(issues)

            # Mark as processed
            self.state.processed_reports[str_path] = file_hash

        return new_issues

    def _parse_report(self, report_path: Path) -> List[Issue]:
        """Parse a markdown report to extract issues."""
        content = report_path.read_text()
        issues = []

        # Pattern for numbered issues with priority markers
        # Matches: "1. **Issue Title** [HIGH]" or "### 1. Issue Title (High Priority)"
        patterns = [
            # Pattern: "### N. Title" with optional priority in brackets
            r'###?\s*(\d+)\.\s*\*?\*?([^*\n\[]+)\*?\*?\s*(?:\[([A-Z]+)\])?',
            # Pattern: "N. **Title**" style
            r'^(\d+)\.\s*\*\*([^*]+)\*\*\s*(?:\[([A-Z]+)\])?',
            # Pattern: "- [ ] Title [PRIORITY]"
            r'^-\s*\[[ x]\]\s*([^[\n]+)\s*\[([A-Z]+)\]',
        ]

        current_priority = IssuePriority.MEDIUM
        lines = content.split('\n')

        for i, line in enumerate(lines):
            # Check for priority section headers
            line_lower = line.lower()
            if 'critical' in line_lower and '#' in line:
                current_priority = IssuePriority.CRITICAL
            elif 'high' in line_lower and '#' in line:
                current_priority = IssuePriority.HIGH
            elif 'medium' in line_lower and '#' in line:
                current_priority = IssuePriority.MEDIUM
            elif 'low' in line_lower and '#' in line:
                current_priority = IssuePriority.LOW

            # Try to match issue patterns
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()

                    # Extract title and priority
                    if len(groups) >= 2:
                        title = groups[1].strip() if groups[1] else groups[0].strip()
                        explicit_priority = groups[2] if len(groups) > 2 and groups[2] else None
                    else:
                        title = groups[0].strip()
                        explicit_priority = groups[1] if len(groups) > 1 else None

                    # Use explicit priority if found
                    priority = current_priority
                    if explicit_priority:
                        try:
                            priority = IssuePriority(explicit_priority.lower())
                        except ValueError:
                            pass

                    # Get description from following lines
                    description_lines = []
                    for j in range(i + 1, min(i + 10, len(lines))):
                        next_line = lines[j].strip()
                        if next_line and not next_line.startswith('#') and not re.match(r'^\d+\.', next_line):
                            description_lines.append(next_line)
                        elif next_line.startswith('#') or re.match(r'^\d+\.', next_line):
                            break

                    description = ' '.join(description_lines)

                    # Extract file path if mentioned
                    file_match = re.search(r'`([^`]+\.(py|js|ts|tsx|jsx))`', description)
                    file_path = file_match.group(1) if file_match else None

                    # Generate unique ID
                    issue_id = hashlib.md5(f"{report_path.name}:{title}".encode()).hexdigest()[:8]

                    # Check if issue already exists
                    existing = next((iss for iss in self.state.issues if iss.id == issue_id), None)
                    if not existing:
                        issue = Issue(
                            id=issue_id,
                            title=title,
                            description=description,
                            priority=priority,
                            file_path=file_path,
                            source_report=str(report_path),
                        )
                        issues.append(issue)
                        logger.info(f"  Found issue: [{priority.value}] {title}")

                    break  # Only match first pattern

        return issues

    def get_next_issue(self) -> Optional[Issue]:
        """Get the next issue to work on."""
        # Priority order
        priority_order = [
            IssuePriority.CRITICAL,
            IssuePriority.HIGH,
            IssuePriority.MEDIUM,
            IssuePriority.LOW,
        ]

        for priority in priority_order:
            for issue in self.state.issues:
                if (issue.status == IssueStatus.PENDING and
                    issue.priority == priority and
                    issue.attempts < issue.max_attempts):
                    return issue

        return None

    def implement_fix(self, issue: Issue) -> bool:
        """Use Claude Code to implement a fix for the issue."""
        logger.info(f"Implementing fix for: {issue.title}")

        issue.status = IssueStatus.IN_PROGRESS
        issue.attempts += 1
        self.state.current_issue_id = issue.id
        self._save_state()

        # Create branch name
        branch_name = f"{self.config.get('branchPrefix', 'compound/')}{issue.id}-{self._slugify(issue.title)}"
        issue.branch_name = branch_name

        # Build the prompt for Claude Code
        prompt = self._build_fix_prompt(issue)

        try:
            # Create branch
            self._run_git(['checkout', '-b', branch_name], check=False)

            # Run Claude Code with the fix prompt
            success = self._run_claude_code(prompt, issue)

            if success:
                # Run quality checks
                if self._run_quality_checks():
                    # Commit and create PR
                    if self._commit_and_pr(issue):
                        issue.status = IssueStatus.COMPLETED
                        issue.completed_at = datetime.now().isoformat()
                        self.state.total_fixes += 1
                        logger.info(f"✓ Successfully fixed: {issue.title}")
                        return True

            # If we get here, something failed
            issue.status = IssueStatus.FAILED if issue.attempts >= issue.max_attempts else IssueStatus.PENDING
            self.state.total_failures += 1
            return False

        except Exception as e:
            logger.error(f"Error implementing fix: {e}")
            issue.last_error = str(e)
            issue.status = IssueStatus.FAILED if issue.attempts >= issue.max_attempts else IssueStatus.PENDING
            self.state.total_failures += 1
            return False

        finally:
            # Always return to main branch
            self._run_git(['checkout', 'main'], check=False)
            self.state.current_issue_id = None
            self._save_state()

    def _build_fix_prompt(self, issue: Issue) -> str:
        """Build prompt for Claude Code to fix the issue."""
        prompt = f"""Fix this issue in the codebase:

## Issue: {issue.title}

{issue.description}

## Instructions:
1. Analyze the issue and identify the root cause
2. Implement the fix with minimal changes
3. Do NOT create new files unless absolutely necessary
4. Do NOT add documentation unless requested
5. Run relevant tests to verify the fix
6. When done, output: COMPOUND_FIX_COMPLETE

## Constraints:
- Keep changes focused and minimal
- Follow existing code patterns
- Do not over-engineer
"""
        if issue.file_path:
            prompt += f"\n## Likely file: {issue.file_path}\n"

        return prompt

    def _run_claude_code(self, prompt: str, issue: Issue) -> bool:
        """Run Claude Code CLI with the given prompt."""
        # Write prompt to temp file
        prompt_file = self.project_root / ".compound_prompt.txt"
        prompt_file.write_text(prompt)

        try:
            # Run Claude Code in non-interactive mode
            result = subprocess.run(
                [
                    'claude',
                    '--print',  # Non-interactive
                    '--dangerously-skip-permissions',  # Auto-approve
                    '-p', prompt,
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per issue
            )

            output = result.stdout + result.stderr

            # Check for completion marker
            if 'COMPOUND_FIX_COMPLETE' in output:
                return True

            # Check for common success patterns
            if any(pattern in output.lower() for pattern in [
                'fix applied', 'changes committed', 'successfully',
                'fixed', 'resolved', 'completed'
            ]):
                return True

            issue.last_error = output[-500:] if len(output) > 500 else output
            return False

        except subprocess.TimeoutExpired:
            issue.last_error = "Claude Code timed out after 10 minutes"
            return False
        except FileNotFoundError:
            issue.last_error = "Claude Code CLI not found. Is it installed?"
            return False
        except Exception as e:
            issue.last_error = str(e)
            return False
        finally:
            prompt_file.unlink(missing_ok=True)

    def _run_quality_checks(self) -> bool:
        """Run configured quality checks."""
        checks = self.config.get('qualityChecks', [])

        for check in checks:
            logger.info(f"Running check: {check}")
            try:
                result = subprocess.run(
                    check,
                    shell=True,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode != 0:
                    logger.warning(f"Check failed: {check}")
                    logger.warning(result.stderr)
                    return False
            except Exception as e:
                logger.warning(f"Check error: {e}")
                return False

        return True

    def _commit_and_pr(self, issue: Issue) -> bool:
        """Commit changes and create PR."""
        # Check for changes
        status = self._run_git(['status', '--porcelain'])
        if not status.strip():
            logger.info("No changes to commit")
            return False

        # Stage all changes
        self._run_git(['add', '-A'])

        # Commit
        commit_msg = f"""fix: {issue.title}

{issue.description[:200] if issue.description else 'Auto-fix by Compound Daemon'}

Issue ID: {issue.id}
Source: {Path(issue.source_report).name if issue.source_report else 'unknown'}

Co-Authored-By: Compound Daemon <compound@continuonxr.local>
"""
        self._run_git(['commit', '-m', commit_msg])

        # Push branch
        self._run_git(['push', '-u', 'origin', issue.branch_name], check=False)

        # Create PR using gh
        try:
            result = subprocess.run(
                [
                    'gh', 'pr', 'create',
                    '--title', f'fix: {issue.title}',
                    '--body', f"""## Summary
- {issue.description[:300] if issue.description else 'Auto-fix'}

## Issue Details
- **ID**: {issue.id}
- **Priority**: {issue.priority.value}
- **Source**: {Path(issue.source_report).name if issue.source_report else 'unknown'}

---
🤖 Generated by Compound Daemon
""",
                    '--base', 'main',
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                # Extract PR number from output
                pr_match = re.search(r'/pull/(\d+)', result.stdout)
                if pr_match:
                    issue.pr_number = int(pr_match.group(1))

                # Auto-merge if enabled
                if self.config.get('autoMerge', True):
                    self._auto_merge_pr(issue.pr_number)

                return True

        except Exception as e:
            logger.error(f"Failed to create PR: {e}")

        return False

    def _auto_merge_pr(self, pr_number: int):
        """Auto-merge a PR."""
        try:
            subprocess.run(
                ['gh', 'pr', 'merge', str(pr_number), '--squash', '--delete-branch'],
                cwd=self.project_root,
                capture_output=True,
                timeout=60,
            )
            logger.info(f"Auto-merged PR #{pr_number}")
        except Exception as e:
            logger.warning(f"Could not auto-merge PR #{pr_number}: {e}")

    def _run_git(self, args: List[str], check: bool = True) -> str:
        """Run a git command."""
        result = subprocess.run(
            ['git'] + args,
            cwd=self.project_root,
            capture_output=True,
            text=True,
        )
        if check and result.returncode != 0:
            raise RuntimeError(f"Git command failed: {result.stderr}")
        return result.stdout

    def _slugify(self, text: str) -> str:
        """Convert text to slug."""
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        slug = re.sub(r'[-\s]+', '-', slug)
        return slug[:30]

    def run_once(self) -> bool:
        """Run a single improvement cycle."""
        logger.info("=" * 50)
        logger.info("Compound Daemon - Starting cycle")
        logger.info("=" * 50)

        # Scan for new issues
        new_issues = self.scan_reports()
        self.state.issues.extend(new_issues)
        self._save_state()

        if new_issues:
            logger.info(f"Found {len(new_issues)} new issues")

        # Get next issue to work on
        issue = self.get_next_issue()

        if not issue:
            logger.info("No pending issues to work on")
            return False

        # Implement the fix
        success = self.implement_fix(issue)
        self.state.last_run = datetime.now().isoformat()
        self._save_state()

        return success

    def run_daemon(self, interval: int = 300):
        """Run continuously as a daemon."""
        logger.info("Compound Daemon starting in continuous mode")
        logger.info(f"Poll interval: {interval} seconds")

        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                logger.info("Daemon stopped by user")
                break
            except Exception as e:
                logger.error(f"Daemon error: {e}")

            # Wait before next cycle
            logger.info(f"Sleeping for {interval} seconds...")
            time.sleep(interval)

    def status(self) -> Dict[str, Any]:
        """Get daemon status."""
        pending = sum(1 for i in self.state.issues if i.status == IssueStatus.PENDING)
        completed = sum(1 for i in self.state.issues if i.status == IssueStatus.COMPLETED)
        failed = sum(1 for i in self.state.issues if i.status == IssueStatus.FAILED)
        in_progress = sum(1 for i in self.state.issues if i.status == IssueStatus.IN_PROGRESS)

        return {
            'total_issues': len(self.state.issues),
            'pending': pending,
            'in_progress': in_progress,
            'completed': completed,
            'failed': failed,
            'total_fixes': self.state.total_fixes,
            'total_failures': self.state.total_failures,
            'last_run': self.state.last_run,
            'started_at': self.state.started_at,
            'reports_processed': len(self.state.processed_reports),
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compound Daemon - Autonomous self-improvement')
    parser.add_argument('--once', action='store_true', help='Run single cycle')
    parser.add_argument('--status', action='store_true', help='Show status')
    parser.add_argument('--interval', type=int, default=300, help='Poll interval in seconds')
    parser.add_argument('--project', type=str, default='.', help='Project root directory')

    args = parser.parse_args()

    project_root = Path(args.project).resolve()
    daemon = CompoundDaemon(project_root)

    if args.status:
        status = daemon.status()
        print("\n📊 Compound Daemon Status")
        print("=" * 40)
        print(f"Total Issues:     {status['total_issues']}")
        print(f"  Pending:        {status['pending']}")
        print(f"  In Progress:    {status['in_progress']}")
        print(f"  Completed:      {status['completed']}")
        print(f"  Failed:         {status['failed']}")
        print(f"Total Fixes:      {status['total_fixes']}")
        print(f"Total Failures:   {status['total_failures']}")
        print(f"Reports Scanned:  {status['reports_processed']}")
        print(f"Last Run:         {status['last_run'] or 'Never'}")
        print(f"Started:          {status['started_at']}")
        return

    if args.once:
        daemon.run_once()
    else:
        daemon.run_daemon(interval=args.interval)


if __name__ == '__main__':
    main()
