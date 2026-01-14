"""
Claude Code CLI Wrapper
=======================

Provides Claude Code CLI integration for the ContinuonBrain chat system.
Uses the local `claude` CLI for agentic AI capabilities with tool use.

This wrapper implements the same interface as ClaudeChat/HopeChat so it can be
used interchangeably as the chat backend, but leverages the full power of
Claude Code's CLI with file access, terminal commands, and web browsing.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatSession:
    """A conversation session with history."""
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    working_directory: str = ""

    def add_message(self, role: str, content: str, **metadata) -> ChatMessage:
        """Add a message to the session."""
        msg = ChatMessage(role=role, content=content, metadata=metadata)
        self.messages.append(msg)
        self.last_activity = time.time()
        return msg

    def get_history(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent history as list of dicts."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages[-limit:]
        ]


class ClaudeCodeCLI:
    """
    Claude Code CLI wrapper for agentic AI chat with full tool capabilities.

    Uses the local `claude` CLI installation for:
    - File read/write/edit operations
    - Terminal command execution
    - Web browsing and search
    - Multi-step task completion

    Implements the same interface as HopeChat/ClaudeChat for drop-in compatibility.
    """

    def __init__(
        self,
        working_directory: Optional[str] = None,
        config_dir: Optional[str] = None,
        model: str = "sonnet",
        max_turns: int = 10,
        timeout: int = 120,
        allowed_tools: Optional[List[str]] = None,
    ):
        """
        Initialize Claude Code CLI wrapper.

        Args:
            working_directory: Working directory for Claude Code operations.
            config_dir: Config directory to load settings from.
            model: Model to use (sonnet, opus, haiku).
            max_turns: Maximum conversation turns per request.
            timeout: Timeout in seconds for CLI execution.
            allowed_tools: List of allowed tools (None = all tools).
        """
        self.model_name = "claude-code"
        self.model = model
        self.config_dir = config_dir
        self.max_turns = max_turns
        self.timeout = timeout
        self.allowed_tools = allowed_tools

        # Default working directory
        if working_directory:
            self.working_directory = Path(working_directory)
        elif config_dir:
            self.working_directory = Path(config_dir).parent
        else:
            self.working_directory = Path.cwd()

        # Find claude CLI
        self.claude_path = self._find_claude_cli()

        # Session management
        self._sessions: Dict[str, ChatSession] = {}
        self._default_session_id = "default"

        # System context for Claude Code
        self.system_context = (
            "You are Claude Code, an agentic AI integrated with ContinuonBrain. "
            "You have access to tools for file operations, terminal commands, and web access. "
            "Help users with coding tasks, system administration, and automation. "
            "Be concise but thorough in your responses."
        )

    def _find_claude_cli(self) -> Optional[str]:
        """Find the claude CLI executable."""
        # Check common locations
        paths_to_check = [
            shutil.which("claude"),
            os.path.expanduser("~/.local/bin/claude"),
            os.path.expanduser("~/.claude/bin/claude"),
            "/usr/local/bin/claude",
            "/usr/bin/claude",
        ]

        for path in paths_to_check:
            if path and os.path.isfile(path) and os.access(path, os.X_OK):
                logger.info(f"Found claude CLI at: {path}")
                return path

        logger.warning("Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code")
        return None

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        system_context: Optional[str] = None,
        working_directory: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a chat message using Claude Code CLI.

        Args:
            message: User's message text.
            session_id: Session ID for conversation tracking.
            history: Optional conversation history (used for context).
            system_context: Optional system prompt override.
            working_directory: Override working directory for this request.

        Returns:
            Dict with response, session_id, and metadata.
        """
        start_time = time.time()

        # Get or create session
        session_id = session_id or self._default_session_id
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatSession(
                session_id=session_id,
                working_directory=str(self.working_directory),
            )
        session = self._sessions[session_id]

        # Add user message to session
        session.add_message("user", message)

        # Check if CLI is available
        if not self.claude_path:
            response_text = (
                "[Error: Claude Code CLI not installed. "
                "Install with: npm install -g @anthropic-ai/claude-code]"
            )
            session.add_message("assistant", response_text)
            return {
                "success": False,
                "response": response_text,
                "session_id": session_id,
                "agent": "claude_code_error",
                "duration_ms": (time.time() - start_time) * 1000,
                "turn_count": len(session.messages),
                "error": "CLI not installed",
            }

        try:
            # Build the CLI command
            work_dir = working_directory or str(self.working_directory)

            # Include conversation context in the prompt
            context_prompt = self._build_context_prompt(
                message,
                history or session.get_history(limit=5),
                system_context,
            )

            # Run claude CLI
            response_text = self._run_claude_cli(
                context_prompt,
                work_dir,
            )

            response_agent = f"claude_code_{self.model}"

        except subprocess.TimeoutExpired:
            logger.error(f"Claude CLI timed out after {self.timeout}s")
            response_text = f"[Claude Code timed out after {self.timeout} seconds. Try a simpler request.]"
            response_agent = "claude_code_timeout"
        except subprocess.CalledProcessError as e:
            logger.error(f"Claude CLI error: {e}")
            response_text = f"[Claude Code error: {e.stderr or str(e)}]"
            response_agent = "claude_code_error"
        except Exception as e:
            logger.error(f"Claude Code CLI error: {e}")
            response_text = f"[Claude Code error: {str(e)}]"
            response_agent = "claude_code_error"

        # Add response to session
        session.add_message("assistant", response_text)

        duration = time.time() - start_time

        return {
            "success": True,
            "response": response_text,
            "session_id": session_id,
            "agent": response_agent,
            "duration_ms": duration * 1000,
            "turn_count": len(session.messages),
            "working_directory": work_dir,
        }

    def _build_context_prompt(
        self,
        message: str,
        history: List[Dict],
        system_context: Optional[str],
    ) -> str:
        """Build the full prompt including context."""
        parts = []

        # Add system context
        if system_context:
            parts.append(f"Context: {system_context}")

        # Add recent history summary if available
        if history and len(history) > 1:
            history_summary = []
            for msg in history[-4:]:  # Last 4 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")[:200]  # Truncate long messages
                if content:
                    history_summary.append(f"{role}: {content}")
            if history_summary:
                parts.append("Recent conversation:\n" + "\n".join(history_summary))

        # Add current message
        parts.append(f"User request: {message}")

        return "\n\n".join(parts)

    def _run_claude_cli(self, prompt: str, work_dir: str) -> str:
        """Execute claude CLI and return the response."""
        # Write prompt to temp file to avoid shell escaping issues
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(prompt)
            prompt_file = f.name

        try:
            # Build command
            cmd = [
                self.claude_path,
                "--print",  # Print-only mode for non-interactive use
                "--model", self.model,
                "--max-turns", str(self.max_turns),
            ]

            # Add allowed tools if specified
            if self.allowed_tools:
                for tool in self.allowed_tools:
                    cmd.extend(["--allowedTools", tool])

            # Run the command
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=work_dir,
                env={**os.environ, "CLAUDE_CODE_HEADLESS": "1"},
            )

            if result.returncode != 0:
                logger.error(f"Claude CLI failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                logger.error(f"STDOUT: {result.stdout[:500] if result.stdout else 'None'}")
                if result.stderr:
                    return f"[Claude Code error: {result.stderr.strip()}]"
                return f"[Claude Code returned error code {result.returncode}]"

            # Parse and clean the output
            output = result.stdout.strip()
            if not output:
                output = "[Claude Code returned no output]"

            return output

        finally:
            # Clean up temp file
            try:
                os.unlink(prompt_file)
            except:
                pass

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def clear_session(self, session_id: str) -> None:
        """Clear a session's history."""
        if session_id in self._sessions:
            self._sessions[session_id].messages.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get chat service status."""
        return {
            "ready": bool(self.claude_path),
            "model": self.model,
            "model_name": self.model_name,
            "cli_path": self.claude_path,
            "cli_installed": bool(self.claude_path),
            "active_sessions": len(self._sessions),
            "working_directory": str(self.working_directory),
            "provider": "claude_code_cli",
        }

    async def chat_async(
        self,
        message: str,
        session_id: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        system_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async version of chat (wraps sync version)."""
        return self.chat(message, session_id, history, system_context)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "name": f"Claude Code CLI ({self.model})",
            "model_id": f"claude-code-{self.model}",
            "provider": "claude_code_cli",
            "type": "local_cli",
            "cli_installed": bool(self.claude_path),
            "working_directory": str(self.working_directory),
            "max_turns": self.max_turns,
        }

    def load_model(self) -> None:
        """Validate CLI availability (called during model switch)."""
        if not self.claude_path:
            # Try to find it again
            self.claude_path = self._find_claude_cli()

        if not self.claude_path:
            raise ValueError(
                "Claude Code CLI not installed. "
                "Install with: npm install -g @anthropic-ai/claude-code"
            )

        logger.info(f"Claude Code CLI ready at: {self.claude_path}")


def create_claude_code_cli(
    config_dir: Optional[str] = None,
    model: str = "sonnet",
    **kwargs,
) -> ClaudeCodeCLI:
    """
    Factory function to create a ClaudeCodeCLI instance.

    Args:
        config_dir: Config directory to load settings from.
        model: Model to use (sonnet, opus, haiku).
        **kwargs: Additional arguments passed to ClaudeCodeCLI.

    Returns:
        ClaudeCodeCLI instance.
    """
    return ClaudeCodeCLI(config_dir=config_dir, model=model, **kwargs)
