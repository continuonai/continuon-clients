"""
Claude Code CLI Integration
============================

Swappable CLI interface for the Meta Ralph Agent.
Supports Claude Code (Opus 4.5), Gemini CLI, Ollama, and custom providers.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, AsyncIterator

logger = logging.getLogger(__name__)


class CLIProvider(Enum):
    """Available CLI providers."""
    CLAUDE_CODE = "claude"       # Claude Code CLI (Opus 4.5)
    GEMINI = "gemini"            # Gemini CLI
    OLLAMA = "ollama"            # Ollama local
    OPENAI = "openai"            # OpenAI API
    CUSTOM = "custom"            # Custom endpoint


@dataclass
class CLIConfig:
    """Configuration for a CLI provider."""
    provider: CLIProvider
    model: str = ""
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: int = 300
    max_tokens: int = 4096
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    extra_args: List[str] = field(default_factory=list)


@dataclass
class CLIResponse:
    """Response from a CLI invocation."""
    text: str
    provider: CLIProvider
    model: str
    latency_ms: float
    tokens_used: Optional[int] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    raw_output: Optional[str] = None


class CLIProviderBase(ABC):
    """Base class for CLI providers."""

    def __init__(self, config: CLIConfig):
        self.config = config

    @abstractmethod
    async def send_message(
        self,
        message: str,
        context: Optional[List[Dict[str, str]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> CLIResponse:
        """Send a message and get a response."""
        pass

    @abstractmethod
    async def stream_message(
        self,
        message: str,
        context: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[str]:
        """Stream a response token by token."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available."""
        pass


class ClaudeCodeProvider(CLIProviderBase):
    """
    Claude Code CLI provider (Opus 4.5).

    Uses the @anthropic-ai/claude-code CLI for interaction.
    """

    def __init__(self, config: CLIConfig):
        super().__init__(config)
        self.cli_command = "claude"
        self.model = config.model or "claude-opus-4-5-20251101"

    async def send_message(
        self,
        message: str,
        context: Optional[List[Dict[str, str]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> CLIResponse:
        """Send a message to Claude Code CLI."""

        start_time = time.time()

        try:
            # Build command
            cmd = [self.cli_command]

            # Add model flag if specified
            if self.model:
                cmd.extend(["--model", self.model])

            # Add system prompt if specified
            if self.config.system_prompt:
                cmd.extend(["--system", self.config.system_prompt])

            # Add extra args
            cmd.extend(self.config.extra_args)

            # Add the message via stdin
            full_message = message
            if context:
                context_str = "\n".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in context
                ])
                full_message = f"Previous context:\n{context_str}\n\nCurrent message: {message}"

            # Run the CLI
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=full_message.encode()),
                timeout=self.config.timeout_seconds
            )

            latency_ms = (time.time() - start_time) * 1000

            if process.returncode != 0:
                return CLIResponse(
                    text="",
                    provider=CLIProvider.CLAUDE_CODE,
                    model=self.model,
                    latency_ms=latency_ms,
                    error=stderr.decode() if stderr else "Unknown error",
                    raw_output=stdout.decode() if stdout else None
                )

            output = stdout.decode()

            # Parse tool calls if present
            tool_calls = self._parse_tool_calls(output)

            return CLIResponse(
                text=output,
                provider=CLIProvider.CLAUDE_CODE,
                model=self.model,
                latency_ms=latency_ms,
                tool_calls=tool_calls,
                raw_output=output
            )

        except asyncio.TimeoutError:
            return CLIResponse(
                text="",
                provider=CLIProvider.CLAUDE_CODE,
                model=self.model,
                latency_ms=(time.time() - start_time) * 1000,
                error="Request timed out"
            )
        except Exception as e:
            return CLIResponse(
                text="",
                provider=CLIProvider.CLAUDE_CODE,
                model=self.model,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

    async def stream_message(
        self,
        message: str,
        context: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[str]:
        """Stream response from Claude Code."""

        cmd = [self.cli_command, "--stream"]

        if self.model:
            cmd.extend(["--model", self.model])

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            if process.stdin:
                process.stdin.write(message.encode())
                await process.stdin.drain()
                process.stdin.close()

            if process.stdout:
                async for line in process.stdout:
                    yield line.decode()

            await process.wait()

        except Exception as e:
            yield f"[Error: {e}]"

    def is_available(self) -> bool:
        """Check if Claude Code CLI is available."""
        try:
            result = subprocess.run(
                [self.cli_command, "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def _parse_tool_calls(self, output: str) -> List[Dict[str, Any]]:
        """Parse tool calls from output."""
        tool_calls = []

        # Look for tool call patterns
        if "<function_calls>" in output:
            # Parse XML-style tool calls
            import re
            pattern = r'<invoke name="([^"]+)">(.*?)</invoke>'
            matches = re.findall(pattern, output, re.DOTALL)
            for name, params in matches:
                tool_calls.append({
                    "name": name,
                    "parameters": params.strip()
                })

        return tool_calls


class GeminiCLIProvider(CLIProviderBase):
    """Gemini CLI provider."""

    def __init__(self, config: CLIConfig):
        super().__init__(config)
        self.cli_command = "gemini"
        self.model = config.model or "gemini-2.0-flash"

    async def send_message(
        self,
        message: str,
        context: Optional[List[Dict[str, str]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> CLIResponse:
        """Send a message to Gemini CLI."""

        start_time = time.time()

        try:
            cmd = [self.cli_command]

            if self.model:
                cmd.extend(["--model", self.model])

            cmd.extend(self.config.extra_args)

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=message.encode()),
                timeout=self.config.timeout_seconds
            )

            latency_ms = (time.time() - start_time) * 1000

            return CLIResponse(
                text=stdout.decode() if stdout else "",
                provider=CLIProvider.GEMINI,
                model=self.model,
                latency_ms=latency_ms,
                error=stderr.decode() if stderr and process.returncode != 0 else None
            )

        except Exception as e:
            return CLIResponse(
                text="",
                provider=CLIProvider.GEMINI,
                model=self.model,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

    async def stream_message(
        self,
        message: str,
        context: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[str]:
        """Stream response from Gemini."""
        # Similar implementation
        response = await self.send_message(message, context)
        yield response.text

    def is_available(self) -> bool:
        """Check if Gemini CLI is available."""
        try:
            result = subprocess.run(
                [self.cli_command, "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False


class OllamaProvider(CLIProviderBase):
    """Ollama local model provider."""

    def __init__(self, config: CLIConfig):
        super().__init__(config)
        self.cli_command = "ollama"
        self.model = config.model or "codellama:13b"
        self.endpoint = config.endpoint or "http://localhost:11434"

    async def send_message(
        self,
        message: str,
        context: Optional[List[Dict[str, str]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> CLIResponse:
        """Send a message to Ollama."""

        start_time = time.time()

        try:
            cmd = [self.cli_command, "run", self.model]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=message.encode()),
                timeout=self.config.timeout_seconds
            )

            latency_ms = (time.time() - start_time) * 1000

            return CLIResponse(
                text=stdout.decode() if stdout else "",
                provider=CLIProvider.OLLAMA,
                model=self.model,
                latency_ms=latency_ms,
                error=stderr.decode() if stderr and process.returncode != 0 else None
            )

        except Exception as e:
            return CLIResponse(
                text="",
                provider=CLIProvider.OLLAMA,
                model=self.model,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

    async def stream_message(
        self,
        message: str,
        context: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[str]:
        """Stream response from Ollama."""
        response = await self.send_message(message, context)
        yield response.text

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            result = subprocess.run(
                [self.cli_command, "list"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False


class ClaudeCodeCLI:
    """
    Unified CLI interface for the Meta Ralph Agent.

    Provides a swappable interface between Claude Code (Opus 4.5),
    Gemini, Ollama, and other providers.
    """

    def __init__(
        self,
        default_provider: CLIProvider = CLIProvider.CLAUDE_CODE,
        config: Optional[CLIConfig] = None
    ):
        self.default_provider = default_provider
        self.config = config or CLIConfig(provider=default_provider)

        # Initialize providers
        self._providers: Dict[CLIProvider, CLIProviderBase] = {}
        self._init_providers()

        # Current active provider
        self._active_provider: Optional[CLIProviderBase] = None
        self._select_provider(default_provider)

        # Conversation history
        self._history: List[Dict[str, str]] = []

        # Callbacks
        self._on_response: List[Callable[[CLIResponse], None]] = []

    def _init_providers(self) -> None:
        """Initialize available providers."""

        providers = [
            (CLIProvider.CLAUDE_CODE, ClaudeCodeProvider),
            (CLIProvider.GEMINI, GeminiCLIProvider),
            (CLIProvider.OLLAMA, OllamaProvider),
        ]

        for provider_type, provider_class in providers:
            try:
                config = CLIConfig(provider=provider_type)
                provider = provider_class(config)
                if provider.is_available():
                    self._providers[provider_type] = provider
                    logger.info(f"Provider available: {provider_type.value}")
            except Exception as e:
                logger.warning(f"Failed to init provider {provider_type.value}: {e}")

    def _select_provider(self, provider: CLIProvider) -> bool:
        """Select the active provider."""

        if provider in self._providers:
            self._active_provider = self._providers[provider]
            logger.info(f"Selected provider: {provider.value}")
            return True

        # Fallback to any available provider
        for p in self._providers.values():
            self._active_provider = p
            logger.warning(f"Fallback to provider: {p.config.provider.value}")
            return True

        logger.error("No providers available")
        return False

    def switch_provider(self, provider: CLIProvider) -> bool:
        """Switch to a different provider."""
        return self._select_provider(provider)

    def get_available_providers(self) -> List[CLIProvider]:
        """Get list of available providers."""
        return list(self._providers.keys())

    def get_active_provider(self) -> Optional[CLIProvider]:
        """Get the currently active provider."""
        if self._active_provider:
            return self._active_provider.config.provider
        return None

    async def send(
        self,
        message: str,
        include_history: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> CLIResponse:
        """
        Send a message to the active CLI provider.
        """
        if not self._active_provider:
            return CLIResponse(
                text="",
                provider=CLIProvider.CLAUDE_CODE,
                model="",
                latency_ms=0,
                error="No provider available"
            )

        # Build context from history
        context = self._history if include_history else None

        # Send message
        response = await self._active_provider.send_message(message, context, tools)

        # Update history
        self._history.append({"role": "user", "content": message})
        if not response.error:
            self._history.append({"role": "assistant", "content": response.text})

        # Notify callbacks
        for callback in self._on_response:
            try:
                callback(response)
            except Exception as e:
                logger.error(f"Response callback error: {e}")

        return response

    async def stream(
        self,
        message: str,
        include_history: bool = True
    ) -> AsyncIterator[str]:
        """
        Stream a response from the active CLI provider.
        """
        if not self._active_provider:
            yield "[Error: No provider available]"
            return

        context = self._history if include_history else None

        # Update history with user message
        self._history.append({"role": "user", "content": message})

        # Stream response
        full_response = ""
        async for chunk in self._active_provider.stream_message(message, context):
            full_response += chunk
            yield chunk

        # Update history with assistant response
        self._history.append({"role": "assistant", "content": full_response})

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self._history.copy()

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for the active provider."""
        if self._active_provider:
            self._active_provider.config.system_prompt = prompt

    def on_response(self, callback: Callable[[CLIResponse], None]) -> None:
        """Subscribe to response events."""
        self._on_response.append(callback)

    # ========== Convenience Methods ==========

    async def ask(self, question: str) -> str:
        """Simple ask method that returns just the text."""
        response = await self.send(question)
        return response.text if not response.error else f"Error: {response.error}"

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> CLIResponse:
        """Execute a tool call through the CLI."""

        # Format as a tool execution request
        tool_request = f"""Execute the following tool:
Tool: {tool_name}
Parameters: {json.dumps(parameters, indent=2)}

Please execute this tool and return the result."""

        return await self.send(tool_request)
