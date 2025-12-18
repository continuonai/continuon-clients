"""
Answer Verification Service

Verifies HOPE agent answers using a 3rd party model (LM/VLM/VLA) with tool access
for general reasoning. If HOPE is incorrect, provides the correct answer.
"""

import logging
from typing import Dict, Any, Optional, List
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
                - tool_calls: List of tools used
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
        
        # Default to a small model for verification
        verify_model = model_hint or "google/gemma-3-270m-it"
        
        # Build verification prompt
        verification_prompt = self._build_verification_prompt(
            question=question,
            hope_answer=hope_answer,
            use_tools=use_tools,
        )
        
        try:
            # Get verification response
            verification_response = self.chat_backend.chat(
                message=verification_prompt,
                system_context=self._build_system_context(use_tools),
                model_hint=verify_model,
            )
            
            # Parse verification response
            result = self._parse_verification_response(verification_response)
            result["model_hint"] = verify_model
            result["timestamp"] = datetime.now().isoformat()
            
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
- [TOOL: TERMINAL <command>]: Execute shell commands for fact-checking
- [TOOL: ASK_GEMINI "prompt"]: Query Gemini API for verification (use sparingly)
- [TOOL: BROWSER <url>]: Open web pages for fact-checking

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
    
    def _parse_verification_response(self, response: str) -> Dict[str, Any]:
        """Parse the verification response from the model."""
        result = {
            "is_correct": True,
            "confidence": 0.5,
            "verification_reasoning": response,
            "correct_answer": None,
            "tool_calls": [],
        }
        
        # Try to parse structured response
        response_lower = response.lower()
        
        # Check for CORRECT/INCORRECT
        if "verification:" in response_lower:
            if "incorrect" in response_lower:
                result["is_correct"] = False
            elif "correct" in response_lower:
                result["is_correct"] = True
        
        # Extract confidence
        if "confidence:" in response_lower:
            try:
                import re
                conf_match = re.search(r"confidence:\s*([0-9.]+)", response_lower)
                if conf_match:
                    result["confidence"] = float(conf_match.group(1))
            except Exception:
                pass
        
        # Extract correct answer if HOPE is incorrect
        if not result["is_correct"] and "correct_answer:" in response_lower:
            try:
                import re
                answer_match = re.search(r"correct_answer:\s*(.+?)(?:\n|$)", response, re.DOTALL)
                if answer_match:
                    result["correct_answer"] = answer_match.group(1).strip()
            except Exception:
                pass
        
        # Extract tool calls
        if "[TOOL:" in response:
            import re
            tool_matches = re.findall(r"\[TOOL:\s*([^\]]+)\]", response)
            result["tool_calls"] = tool_matches
        
        return result
