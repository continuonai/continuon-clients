"""
Network Gate - Enforces sandbox rules on API/network access.

This is the robot equivalent of Cowork's network proxy.
All external API calls go through this gate, which enforces
domain allow-lists and logs all access attempts.
"""

from typing import Any, Callable
from urllib.parse import urlparse
import time


from .manager import Sandbox, SandboxViolation


class NetworkGate:
    """
    Gate that enforces sandbox rules on network access.

    All API calls (LLM, cloud services, etc.) must go through
    this gate, which enforces domain allow-lists.

    Usage:
        sandbox = Sandbox("agent_1", SandboxConfig())
        gate = NetworkGate(sandbox)

        # Check before making any API call
        gate.check_url("https://api.anthropic.com/v1/messages")

        # Or wrap an HTTP client
        safe_client = gate.wrap_client(httpx.Client())
    """

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        self._audit_log: list[dict] = []

    def check_url(self, url: str) -> bool:
        """
        Check if a URL is allowed.

        Extracts the domain and checks against sandbox rules.
        """
        entry = {
            "timestamp": time.time(),
            "type": "network_check",
            "url": url,
        }

        try:
            # Parse domain from URL
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path.split("/")[0]

            # Remove port if present
            if ":" in domain:
                domain = domain.split(":")[0]

            entry["domain"] = domain

            # Check sandbox permissions
            self.sandbox.check_network(domain)

            # Log success
            entry["allowed"] = True
            self._audit_log.append(entry)

            return True

        except SandboxViolation as e:
            # Log failure
            entry["allowed"] = False
            entry["violation"] = str(e)
            entry["violation_type"] = e.violation_type
            self._audit_log.append(entry)
            raise

    def check_domain(self, domain: str) -> bool:
        """Check if a domain is allowed (without URL parsing)."""
        entry = {
            "timestamp": time.time(),
            "type": "domain_check",
            "domain": domain,
        }

        try:
            self.sandbox.check_network(domain)

            entry["allowed"] = True
            self._audit_log.append(entry)
            return True

        except SandboxViolation as e:
            entry["allowed"] = False
            entry["violation"] = str(e)
            entry["violation_type"] = e.violation_type
            self._audit_log.append(entry)
            raise

    def wrap_request(
        self,
        request_func: Callable[..., Any],
        url: str,
        *args,
        **kwargs,
    ) -> Any:
        """
        Wrap a request function with sandbox checks.

        Usage:
            result = gate.wrap_request(
                requests.get,
                "https://api.anthropic.com/v1/messages",
                headers={"Authorization": "Bearer ..."}
            )
        """
        self.check_url(url)
        return request_func(url, *args, **kwargs)

    def create_guarded_caller(
        self,
        api_func: Callable[..., Any],
        domain: str,
    ) -> Callable[..., Any]:
        """
        Create a guarded version of an API function.

        Usage:
            # Original
            response = anthropic_client.messages.create(...)

            # Guarded
            guarded_create = gate.create_guarded_caller(
                anthropic_client.messages.create,
                "api.anthropic.com"
            )
            response = guarded_create(...)
        """

        def guarded(*args, **kwargs):
            self.check_domain(domain)
            return api_func(*args, **kwargs)

        return guarded

    def get_audit_log(self) -> list[dict]:
        """Get the network access audit log."""
        return self._audit_log.copy()

    def get_stats(self) -> dict:
        """Get network gate statistics."""
        total = len(self._audit_log)
        allowed = sum(1 for e in self._audit_log if e.get("allowed", False))
        denied = total - allowed

        # Count by domain
        domains: dict[str, int] = {}
        for entry in self._audit_log:
            domain = entry.get("domain", "unknown")
            domains[domain] = domains.get(domain, 0) + 1

        return {
            "total_requests": total,
            "allowed": allowed,
            "denied": denied,
            "domains": domains,
        }


class GatedLLMClient:
    """
    A simple LLM client wrapper that enforces sandbox rules.

    This provides a convenient way to use LLM APIs while
    respecting sandbox network restrictions.

    Usage:
        gate = NetworkGate(sandbox)
        llm = GatedLLMClient(gate)

        # This will check domain permissions before calling
        response = llm.call_anthropic(messages=[...])
    """

    def __init__(self, network_gate: NetworkGate):
        self.gate = network_gate

    def call_anthropic(
        self,
        messages: list[dict],
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        **kwargs,
    ) -> dict | None:
        """Call Anthropic API (if allowed)."""
        try:
            self.gate.check_domain("api.anthropic.com")
        except SandboxViolation:
            return None

        # Import here to avoid hard dependency
        try:
            import anthropic

            client = anthropic.Anthropic()
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                **kwargs,
            )
            return {
                "content": response.content[0].text,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            }
        except ImportError:
            return {"error": "anthropic package not installed"}
        except Exception as e:
            return {"error": str(e)}

    def call_openai(
        self,
        messages: list[dict],
        model: str = "gpt-4o",
        max_tokens: int = 1024,
        **kwargs,
    ) -> dict | None:
        """Call OpenAI API (if allowed)."""
        try:
            self.gate.check_domain("api.openai.com")
        except SandboxViolation:
            return None

        try:
            import openai

            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                **kwargs,
            )
            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                },
            }
        except ImportError:
            return {"error": "openai package not installed"}
        except Exception as e:
            return {"error": str(e)}

    def call_local(
        self,
        messages: list[dict],
        model: str = "llama3",
        **kwargs,
    ) -> dict | None:
        """
        Call local Ollama API.

        Local calls (localhost) are always allowed.
        """
        # Localhost is implicitly allowed
        try:
            import requests

            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    **kwargs,
                },
                timeout=60,
            )
            data = response.json()
            return {
                "content": data.get("message", {}).get("content", ""),
                "model": model,
            }
        except ImportError:
            return {"error": "requests package not installed"}
        except Exception as e:
            return {"error": str(e)}
