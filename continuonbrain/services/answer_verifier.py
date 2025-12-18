"""
Answer Verification Service

Verifies HOPE agent answers using a 3rd party model (LM/VLM/VLA) with tool access
for general reasoning. If HOPE is incorrect, provides the correct answer.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Callable, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class AnswerVerifier:
    """
    Verifies HOPE agent answers using a 3rd party model with tool access.
    """
    
    def __init__(self, chat_backend, config_dir: Optional[str] = None):
        """
        Initialize the verifier.
        
        Args:
            chat_backend: Chat backend for 3rd party model (Gemma, Phi-2, etc.)
            config_dir: Configuration directory for logging
        """
        self.chat_backend = chat_backend
        self.config_dir = config_dir
        
    def verify_answer(
        self,
        question: str,
        hope_answer: str,
        model_hint: Optional[str] = None,
        use_tools: bool = True,
    ) -> Dict[str, Any]:
        """
        Verify HOPE's answer using a 3rd party model.
        
        Args:
            question: Original user question
            hope_answer: HOPE agent's answer
            model_hint: Model to use for verification (default: gemma-3-270m-it)
            use_tools: Whether to allow tool use for reasoning

        Returns:
            Dict with:
                - is_correct: bool
                - confidence: float
                - verification_reasoning: str
                - correct_answer: Optional[str] (if HOPE is incorrect)
                - tool_calls: List of structured tool invocations (with args/results)
        """
        if not self.chat_backend:
            logger.warning("No chat backend available for verification")
            return {
                "is_correct": True,  # Assume correct if can't verify
                "confidence": 0.0,
                "verification_reasoning": "Verification unavailable - no model backend",
                "correct_answer": None,
                "tool_calls": [],
            }
        
        # Default to a small model for verification (function-calling capable when tools are enabled)
        verify_model = model_hint or (
            "functiongemma-270m-it" if use_tools else "google/gemma-3-270m-it"
        )

        # Build verification prompt
        verification_prompt = self._build_verification_prompt(
            question=question,
            hope_answer=hope_answer,
            use_tools=use_tools,
        )

        tools_spec: List[Dict[str, Any]] = []
        tool_registry: Dict[str, Callable[..., Any]] = {}
        if use_tools:
            tools_spec, tool_registry = self._build_tool_registry()

        try:
            verification_response = self._request_verification(
                verification_prompt,
                verify_model,
                use_tools,
                tools_spec,
            )

            # Parse verification response
            result = self._parse_verification_response(
                verification_response,
                tool_registry=tool_registry,
            )
            result["model_hint"] = verify_model
            result["timestamp"] = datetime.now().isoformat()
            if use_tools:
                result["tools_available"] = [t.get("name") for t in tools_spec]

            return result

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {
                "is_correct": True,  # Assume correct on error
                "confidence": 0.0,
                "verification_reasoning": f"Verification error: {str(e)}",
                "correct_answer": None,
                "tool_calls": [],
                "error": str(e),
            }
    
    def _build_verification_prompt(
        self,
        question: str,
        hope_answer: str,
        use_tools: bool,
    ) -> str:
        """Build the verification prompt for the 3rd party model."""
        prompt = f"""You are a verification assistant. Your task is to verify whether HOPE agent's answer to a question is correct.

QUESTION: {question}

HOPE AGENT'S ANSWER: {hope_answer}

Please:
1. Evaluate if HOPE's answer is factually correct
2. Use tools if needed to verify facts (web search, knowledge base, etc.)
3. If HOPE is incorrect, provide the correct answer
4. Explain your reasoning

Format your response as:
VERIFICATION: [CORRECT|INCORRECT]
CONFIDENCE: [0.0-1.0]
REASONING: [Your reasoning process]
"""
        
        if use_tools:
            prompt += """
TOOLS AVAILABLE:
- terminal(command: str): Execute a bounded shell command for fact-checking (echo/uname/read-only file queries)
- query_status(): Lightweight system query helper for context (returns uptime + working directory)

Use tools to verify facts when needed.
"""
        
        prompt += """
CORRECT_ANSWER: [Only provide if HOPE is INCORRECT, otherwise leave blank]

Begin verification:"""
        
        return prompt
    
    def _build_system_context(self, use_tools: bool) -> str:
        """Build system context for verification."""
        context = """You are a fact-checking and reasoning assistant. Your role is to verify answers from the HOPE agent using:
- Factual knowledge
- Logical reasoning
- Tool access for verification
- Clear explanations

Be thorough but concise. If HOPE is correct, acknowledge it. If incorrect, provide the correct answer clearly."""
        
        if use_tools:
            context += "\n\nYou have access to tools for fact-checking. Use them when needed to verify information."
        
        return context
    
    def _parse_verification_response(
        self,
        response: Any,
        *,
        tool_registry: Optional[Dict[str, Callable[..., Any]]] = None,
    ) -> Dict[str, Any]:
        """Parse the verification response from the model."""
        result = {
            "is_correct": True,
            "confidence": 0.5,
            "verification_reasoning": str(response),
            "correct_answer": None,
            "tool_calls": [],
        }

        text_response = ""
        tool_calls_raw: List[Dict[str, Any]] = []
        if isinstance(response, dict):
            text_response = str(
                response.get("text")
                or response.get("response")
                or response.get("output")
                or ""
            )
            tool_calls_raw = response.get("tool_calls") or []
            if "is_correct" in response:
                result["is_correct"] = bool(response.get("is_correct"))
            if "confidence" in response:
                try:
                    result["confidence"] = float(response.get("confidence"))
                except Exception:
                    pass
            if response.get("correct_answer"):
                result["correct_answer"] = str(response.get("correct_answer"))
            result["raw_response"] = response
        else:
            text_response = str(response)

        result["verification_reasoning"] = text_response or result["verification_reasoning"]

        # Try to parse structured response from text when no explicit fields are provided
        response_lower = text_response.lower()

        # Check for CORRECT/INCORRECT
        if "verification:" in response_lower:
            if "incorrect" in response_lower:
                result["is_correct"] = False
            elif "correct" in response_lower:
                result["is_correct"] = True
        
        # Extract confidence if the model did not provide one directly
        if "confidence:" in response_lower and result.get("confidence") == 0.5:
            try:
                import re
                conf_match = re.search(r"confidence:\s*([0-9.]+)", response_lower)
                if conf_match:
                    result["confidence"] = float(conf_match.group(1))
            except Exception:
                pass
        
        # Extract correct answer if HOPE is incorrect
        if not result["is_correct"] and "correct_answer:" in response_lower and not result.get("correct_answer"):
            try:
                import re
                answer_match = re.search(
                    r"correct_answer:\s*(.+?)(?:\n|$)",
                    text_response or str(response),
                    re.DOTALL,
                )
                if answer_match:
                    result["correct_answer"] = answer_match.group(1).strip()
            except Exception:
                pass

        executed_calls = self._execute_tool_calls(tool_calls_raw, tool_registry)
        if executed_calls:
            result["tool_calls"] = executed_calls

        return result

    def _request_verification(
        self,
        verification_prompt: str,
        model_hint: str,
        use_tools: bool,
        tools_spec: List[Dict[str, Any]],
    ) -> Any:
        """
        Dispatch a verification request, preferring structured tool outputs when available.
        """

        system_context = self._build_system_context(use_tools)
        if use_tools:
            # Prefer a structured tool-calling backend if exposed by the chat backend.
            for attr in ("chat_with_tools", "chat_structured"):
                handler = getattr(self.chat_backend, attr, None)
                if callable(handler):
                    return handler(
                        message=verification_prompt,
                        system_context=system_context,
                        model_hint=model_hint,
                        tools=tools_spec,
                        use_tools=True,
                    )

        # Fallback to plain chat() without tool binding
        return self.chat_backend.chat(
            message=verification_prompt,
            system_context=system_context,
            model_hint=model_hint,
        )

    def _build_tool_registry(self) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., Any]]]:
        """Create a minimal tool registry for fact-checking runs."""

        def terminal(command: str) -> Dict[str, Any]:
            import shlex
            import subprocess

            allowed_prefixes = ("echo", "uname", "cat", "head", "tail")
            parts = shlex.split(command)
            if not parts or parts[0] not in allowed_prefixes:
                raise ValueError("Command not permitted for verification")

            completed = subprocess.run(
                parts,
                check=False,
                capture_output=True,
                text=True,
                timeout=8,
            )
            return {
                "stdout": completed.stdout[:4000],
                "stderr": completed.stderr[:2000],
                "returncode": completed.returncode,
            }

        def query_status() -> Dict[str, Any]:
            import os
            import time

            return {
                "cwd": os.getcwd(),
                "unix_time": int(time.time()),
            }

        tools = [
            {
                "name": "terminal",
                "description": "Run a bounded, read-only shell command (echo, uname, read small files).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to execute"},
                    },
                    "required": ["command"],
                },
            },
            {
                "name": "query_status",
                "description": "Retrieve a small status snippet (cwd and current unix time).",
                "parameters": {"type": "object", "properties": {}},
            },
        ]
        registry = {
            "terminal": terminal,
            "query_status": query_status,
        }
        return tools, registry

    def _execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_registry: Optional[Dict[str, Callable[..., Any]]],
    ) -> List[Dict[str, Any]]:
        if not tool_calls:
            return []

        executed: List[Dict[str, Any]] = []
        registry = tool_registry or {}
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else None
            record = {
                "name": name,
                "arguments": call.get("arguments") if isinstance(call, dict) else None,
                "ok": False,
            }

            # Preserve backend-supplied results if present
            if isinstance(call, dict) and call.get("result") is not None:
                record["result"] = call.get("result")
                record["ok"] = call.get("ok", True)
                executed.append(record)
                continue

            if name in registry and callable(registry[name]):
                args = call.get("arguments") if isinstance(call, dict) else {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"command": args}
                try:
                    result = registry[name](**(args or {}))
                    record["result"] = result
                    record["ok"] = True
                except Exception as exc:  # noqa: BLE001
                    record["error"] = str(exc)
            else:
                record["error"] = "Tool not available"

            executed.append(record)
        return executed
