"""
Ralph Layer Configuration
=========================

Loads configuration from environment variables for the Ralph layer system.
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from .claude_code_cli import CLIProvider, CLIConfig


@dataclass
class RalphLayerConfig:
    """Configuration for the Ralph Layer system."""

    # CLI Provider Settings
    default_cli_provider: CLIProvider = CLIProvider.CLAUDE_CODE
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    ollama_endpoint: str = "http://localhost:11434"

    # Context Management
    context_window_tokens: int = 32768
    max_iterations: int = 100

    # Guardrails
    enable_guardrails: bool = True

    @classmethod
    def from_environment(cls) -> "RalphLayerConfig":
        """Load configuration from environment variables."""

        # Parse provider
        provider_str = os.getenv("RALPH_DEFAULT_CLI_PROVIDER", "claude")
        provider_map = {
            "claude": CLIProvider.CLAUDE_CODE,
            "gemini": CLIProvider.GEMINI,
            "ollama": CLIProvider.OLLAMA,
            "openai": CLIProvider.OPENAI,
        }
        default_provider = provider_map.get(provider_str, CLIProvider.CLAUDE_CODE)

        return cls(
            # Provider settings
            default_cli_provider=default_provider,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            ollama_endpoint=os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"),

            # Context settings
            context_window_tokens=int(os.getenv("RALPH_CONTEXT_WINDOW_TOKENS", "32768")),
            max_iterations=int(os.getenv("RALPH_MAX_ITERATIONS", "100")),

            # Guardrails
            enable_guardrails=os.getenv("RALPH_ENABLE_GUARDRAILS", "1") == "1",
        )

    def get_cli_config(self, provider: Optional[CLIProvider] = None) -> CLIConfig:
        """Get CLI configuration for a specific provider."""

        provider = provider or self.default_cli_provider

        api_key = None
        model = ""
        endpoint = None

        if provider == CLIProvider.CLAUDE_CODE:
            api_key = self.anthropic_api_key
            model = "claude-opus-4-5-20251101"
        elif provider == CLIProvider.GEMINI:
            api_key = self.gemini_api_key
            model = "gemini-2.0-flash"
        elif provider == CLIProvider.OPENAI:
            api_key = self.openai_api_key
            model = "gpt-4o"
        elif provider == CLIProvider.OLLAMA:
            endpoint = self.ollama_endpoint
            model = "codellama:13b"

        return CLIConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            endpoint=endpoint,
            max_tokens=4096,
        )

    def has_provider_credentials(self, provider: CLIProvider) -> bool:
        """Check if credentials are configured for a provider."""

        if provider == CLIProvider.CLAUDE_CODE:
            return self.anthropic_api_key is not None
        elif provider == CLIProvider.GEMINI:
            return self.gemini_api_key is not None
        elif provider == CLIProvider.OPENAI:
            return self.openai_api_key is not None
        elif provider == CLIProvider.OLLAMA:
            # Ollama doesn't need API key
            return True

        return False

    def get_available_providers(self) -> list[CLIProvider]:
        """Get list of providers with valid credentials."""
        return [p for p in CLIProvider if self.has_provider_credentials(p)]


# Global configuration instance (loaded on import)
_config: Optional[RalphLayerConfig] = None


def get_ralph_config() -> RalphLayerConfig:
    """Get the global Ralph configuration, loading from environment if needed."""
    global _config
    if _config is None:
        _config = RalphLayerConfig.from_environment()
    return _config


def reload_ralph_config() -> RalphLayerConfig:
    """Force reload configuration from environment."""
    global _config
    _config = RalphLayerConfig.from_environment()
    return _config
