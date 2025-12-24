"""
Gemma 3 Nano chat interface for robot control.

This module provides local Gemma inference for conversational AI
integrated with robot control and vision capabilities.
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def _resolve_hf_hub_dir() -> Path:
    """
    Resolve the HuggingFace Hub cache directory.

    Priority:
    - HUGGINGFACE_HUB_CACHE (explicit)
    - HF_HOME/hub
    - ~/.cache/huggingface/hub
    - fallback to /home/craigm26/.cache/huggingface/hub (common on this device)
      so root-run services can still use the pre-populated cache without downloading.
    """
    env_hub = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env_hub:
        return Path(env_hub)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    default = Path.home() / ".cache" / "huggingface" / "hub"
    if default.exists():
        return default
    shared = Path("/home/craigm26/.cache/huggingface/hub")
    return shared

def _allow_model_downloads() -> bool:
    return os.environ.get("CONTINUON_ALLOW_MODEL_DOWNLOADS", "0").lower() in ("1", "true", "yes", "on")

def _ensure_hf_cache_env() -> Path:
    """
    Ensure HuggingFace cache env points at our resolved cache location.

    This matters for services running under different users (e.g., systemd root/continuon)
    that should still reuse a pre-populated shared cache.
    """
    hub_dir = _resolve_hf_hub_dir()
    # HuggingFace hub honors HUGGINGFACE_HUB_CACHE directly.
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_dir))
    return hub_dir

def _snapshot_download_path(model_id: str, hf_token: Optional[str]) -> Path:
    """
    Return a local snapshot path for a model.

    - Tries local cache only first (offline-first).
    - If CONTINUON_ALLOW_MODEL_DOWNLOADS=1, allows downloading missing files.
    """
    _ensure_hf_cache_env()
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"huggingface_hub not available: {exc}") from exc

    try:
        return Path(
            snapshot_download(
                repo_id=model_id,
                local_files_only=True,
                token=hf_token,
            )
        )
    except Exception as exc:
        if not _allow_model_downloads():
            raise RuntimeError(
                f"Model {model_id} not available in local cache ({_resolve_hf_hub_dir()}); "
                "downloads disabled (CONTINUON_ALLOW_MODEL_DOWNLOADS=0)."
            ) from exc
        return Path(
            snapshot_download(
                repo_id=model_id,
                local_files_only=False,
                token=hf_token,
            )
        )

def _model_in_local_hf_cache(model_id: str) -> bool:
    try:
        _snapshot_download_path(model_id, hf_token=os.environ.get("HUGGINGFACE_TOKEN"))
        return True
    except Exception:
        return False


class FunctionGemmaChat:
    """
    Lightweight client for google/functiongemma-270m-it with tool-calling support.

    Uses HuggingFace Inference API when available and falls back to a local
    transformers pipeline (offline-first) that respects the shared HF cache.
    """

    MODEL_ID = "google/functiongemma-270m-it"

    def __init__(self, *, hf_token: Optional[str] = None) -> None:
        self.hf_token = hf_token or os.environ.get("HUGGINGFACE_TOKEN")
        self.model_id = self.MODEL_ID
        self.client = None
        self.pipeline = None
        self.device = os.environ.get("FUNCTIONGEMMA_DEVICE", "auto")
        self._init_inference_client()

    def _init_inference_client(self) -> None:
        try:
            from huggingface_hub import InferenceClient

            self.client = InferenceClient(token=self.hf_token)
        except Exception as exc:  # noqa: BLE001
            logger.info(f"InferenceClient unavailable, will fall back to local pipeline ({exc})")
            self.client = None

    def _ensure_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline
        try:
            from transformers import pipeline
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"transformers required for FunctionGemma pipeline: {exc}") from exc

        snapshot_path = _snapshot_download_path(self.model_id, hf_token=self.hf_token)
        self.pipeline = pipeline(
            "text-generation",
            model=str(snapshot_path),
            tokenizer=str(snapshot_path),
            trust_remote_code=True,
            device_map=self.device,
        )
        return self.pipeline

    def _prepare_messages(self, message: str, system_context: Optional[str]) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_context:
            messages.append({"role": "system", "content": system_context})
        messages.append({"role": "user", "content": message})
        return messages

    def _render_tool_calls(self, choice: Any) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        tool_calls = getattr(choice, "tool_calls", None)
        if not tool_calls:
            return calls
        for call in tool_calls:
            fn = getattr(call, "function", None)
            calls.append(
                {
                    "id": getattr(call, "id", None),
                    "type": getattr(call, "type", None),
                    "name": getattr(fn, "name", None) if fn else None,
                    "arguments": getattr(fn, "arguments", None) if fn else None,
                }
            )
        return calls

    def chat(
        self,
        message: str,
        *,
        system_context: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,  # noqa: ARG002 - reserved
        **_: Any,
    ) -> Dict[str, Any]:
        messages = self._prepare_messages(message, system_context)

        if self.client:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    tools=tools or None,
                    tool_choice="auto" if tools else None,
                    max_tokens=512,
                    temperature=0.2,
                )
                choice = completion.choices[0].message
                return {
                    "text": getattr(choice, "content", "") or "",
                    "tool_calls": self._render_tool_calls(choice),
                    "raw": completion.model_dump(exclude_none=True) if hasattr(completion, "model_dump") else None,
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"FunctionGemma inference client failed; falling back to pipeline ({exc})")

        pipe = self._ensure_pipeline()
        outputs = pipe(
            messages=messages,
            tools=tools,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.2,
            return_full_text=False,
        )

        text: str = ""
        tool_calls: List[Dict[str, Any]] = []
        try:
            if isinstance(outputs, list) and outputs:
                generated = outputs[0].get("generated_text") if isinstance(outputs[0], dict) else outputs[0]
                if isinstance(generated, list) and generated:
                    # Chat template style output
                    last = generated[-1]
                    text = last.get("content", "") if isinstance(last, dict) else str(last)
                    tool_calls = last.get("tool_calls", []) if isinstance(last, dict) else []
                elif isinstance(generated, dict):
                    text = generated.get("content", "")
                    tool_calls = generated.get("tool_calls", []) or []
                else:
                    text = str(generated)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to parse FunctionGemma pipeline output ({exc}); using raw text fallback")
            text = str(outputs)

        return {"text": text or "", "tool_calls": tool_calls, "raw": outputs}


class GemmaChat:
    """
    Manages Gemma 3 Nano model inference for chat interactions.
    
    For production deployment, this uses HuggingFace transformers library
    with quantized models for efficient on-device inference.
    """
    DEFAULT_MODEL_ID = "google/gemma-3-270m-it"  # Use model that's actually in cache
    # Fallbacks (retain prior defaults for larger variants):
    # "google/gemma-3n-E2B-it"

    def __init__(self, model_name: str = DEFAULT_MODEL_ID, device: str = "cpu", api_base: Optional[str] = None, api_key: Optional[str] = None, accelerator_device: Optional[str] = None):
        """
        Initialize Gemma chat interface.
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cpu', 'cuda', or 'mps' for Apple Silicon
        api_base: Base URL for OpenAI-compatible API (e.g., vLLM)
            api_key: API key for OpenAI-compatible API
            accelerator_device: Optional detected AI accelerator (e.g., 'hailo8l', 'tpu')
        """
        self.model_name = model_name
        self.device = device
        self.api_base = api_base
        self.api_key = api_key or "EMPTY"  # Default for local vLLM/llama.cpp
        
        self.accelerator_device = accelerator_device
        if self.accelerator_device:
            logger.info(f"Initializing GemmaChat with accelerator: {self.accelerator_device}")
        
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.is_vlm = False
        self.client = None
        self._function_chat: Optional[FunctionGemmaChat] = None
        
        self.chat_history: List[Dict[str, str]] = []
        self.max_history = 10  # Keep last 10 turns
        
        # Check for HuggingFace token
        self.hf_token = os.environ.get('HUGGINGFACE_TOKEN')
        if not self.hf_token and not self.api_base:
            logger.warning("HUGGINGFACE_TOKEN not set - gated models will not be accessible")
            
        if self.api_base:
            try:
                from openai import OpenAI
                self.client = OpenAI(base_url=self.api_base, api_key=self.api_key)
                logger.info(f"Initialized OpenAI client pointing to {self.api_base}")
            except ImportError:
                logger.error("openai package not installed. Cannot use remote API.")

    def load_model(self) -> bool:
        """
        Load Gemma model and tokenizer (or verify API connection).
        
        Returns:
            True if successful, False otherwise
        """
        if self.client:
            # For API, we just assume it's ready or do a quick check?
            # Keeping it simple: if client is initialized, we are "loaded"
            return True

        try:
            # Try importing transformers
            # Try importing transformers
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
            import torch
            try:
                from transformers import AutoModelForImageTextToText
            except ImportError:
                try:
                    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText
                except ImportError:
                    AutoModelForImageTextToText = None
            
            logger.info(f"Loading Gemma model: {self.model_name}")
            # Resolve snapshot path (local cache first).
            snapshot_path = None
            try:
                snapshot_path = _snapshot_download_path(self.model_name, self.hf_token)
                logger.info(f"✅ Using HuggingFace snapshot: {snapshot_path}")
            except Exception as exc:
                logger.error(f"Failed to resolve local snapshot for {self.model_name}: {exc}")
                return False
            
            # --- VLM Loading Logic (Gemma 3N / PaliGemma / Gemma 3) ---
            if "gemma-3" in self.model_name or "paligemma" in self.model_name:
                try:
                    logger.info(f"Attempting to load VLM: {self.model_name}")
                    
                    if AutoModelForImageTextToText is None:
                        raise ImportError("AutoModelForImageTextToText not available in installed transformers version")
                        
                    # We already resolved snapshot_path above.
                    if not snapshot_path:
                        raise RuntimeError("No snapshot path available for VLM load")
                    
                    # Load processor - ALWAYS try local first
                    try:
                        logger.info("Loading processor from snapshot (local_files_only=True)")
                        self.processor = AutoProcessor.from_pretrained(
                            str(snapshot_path),
                            local_files_only=True,
                            trust_remote_code=True,
                        )
                        logger.info("✅ Processor loaded")
                    except Exception as e:
                        logger.error(f"Failed to load processor from snapshot: {e}")
                        raise
                    
                    # Determine dtype and device placement.
                    # NOTE: Some transformer configs + low_cpu_mem_usage can leave parameters on the "meta" device
                    # and crash with "Cannot copy out of meta tensor". For CPU, prefer a plain load without device_map.
                    dtype = torch.bfloat16
                    device_map = None
                    low_cpu_mem_usage = False
                    
                    # Load model - ALWAYS use local cache if available
                    try:
                        logger.info("Loading VLM model from snapshot (local_files_only=True)")
                        self.model = AutoModelForImageTextToText.from_pretrained(
                            str(snapshot_path),
                            local_files_only=True,
                            trust_remote_code=True,
                            device_map=device_map,
                            torch_dtype=dtype,
                            low_cpu_mem_usage=low_cpu_mem_usage,
                        )
                        logger.info("✅ VLM model loaded")
                    except Exception as e:
                        # If local load failed, try a more conservative CPU-only retry with
                        # different settings than the initial attempt (mirrors CausalLM fallback).
                        retry_dtype = None
                        retry_low_cpu_mem_usage = True
                        logger.warning(
                            f"VLM load failed: {e}. Retrying on CPU with torch_dtype=None, low_cpu_mem_usage=True..."
                        )
                        self.model = AutoModelForImageTextToText.from_pretrained(
                            str(snapshot_path),
                            local_files_only=True,
                            trust_remote_code=True,
                            device_map=None,
                            torch_dtype=retry_dtype,
                            low_cpu_mem_usage=retry_low_cpu_mem_usage,
                        )
                        logger.info("✅ VLM model loaded on CPU")
                    
                    self.is_vlm = True
                    logger.info(" Gemma 3N VLM loaded successfully!")
                    return True
                    
                except Exception as e:
                    # Specific check for configuration class errors (e.g. Gemma3TextConfig vs AutoModelForImageTextToText)
                    if "Unrecognized configuration class" in str(e) and "gemma-3" in self.model_name.lower():
                         logger.warning("Caught unrecognized config for VLM load. This is expected for Gemma 3 text models.")
                    else:
                         logger.warning(f"Failed to load as VLM ({e}). Falling back to CausalLM (text-only)...")
                    # Clear partial loads
                    self.model = None
                    self.processor = None

            # --- CausalLM Loading Logic (Text Only / Fallback) ---
            if not snapshot_path:
                raise RuntimeError("No snapshot path available for CausalLM load")
            
            # Load tokenizer - ALWAYS try local first, never download if cached
            try:
                logger.info("Loading tokenizer from snapshot (local_files_only=True)")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(snapshot_path),
                    local_files_only=True,
                    trust_remote_code=True,
                )
                logger.info("✅ Tokenizer loaded")
            except Exception as e:
                logger.error(f"Failed to load tokenizer from snapshot: {e}")
                raise
            
            # Load model with conservative defaults to avoid disk_offload errors on CPU
            use_device_map = None
            torch_dtype = None
            low_cpu_mem = True

            if self.device != "cpu":
                use_device_map = "auto"
                torch_dtype = None  # let transformers pick best for accel
                low_cpu_mem = False
            else:
                torch_dtype = None  # stay with default fp32 on CPU

            # Explicitly check for accelerate; disable auto map if missing
            try:
                import accelerate
                logger.info(f"Accelerate available: {accelerate.__version__}")
            except ImportError:
                if use_device_map == "auto":
                    logger.warning("Accelerate not found. Disabling device_map='auto'")
                    use_device_map = None

            # ALWAYS use local cache if available - never download
            try:
                logger.info("Loading CausalLM model from snapshot (local_files_only=True)")
                logger.info(f"  Snapshot: {snapshot_path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(snapshot_path),
                    local_files_only=True,
                    trust_remote_code=True,
                    device_map=use_device_map,
                    low_cpu_mem_usage=low_cpu_mem,
                    torch_dtype=torch_dtype,
                )
                logger.info("✅ Model loaded")
            except Exception as e:
                # If local load failed, try CPU-only retry (no device_map)
                logger.warning(f"CausalLM load failed: {e}. Retrying on CPU...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(snapshot_path),
                    local_files_only=True,
                    trust_remote_code=True,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    torch_dtype=None,
                )
                logger.info("✅ Model loaded on CPU")
            
            self.is_vlm = False
            logger.info("Gemma model loaded (CausalLM mode)")
            return True
            
        except ImportError as e:
            logger.error(f"transformers library or dependency missing: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load Gemma model: {e}")
            return False
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        if self.client:
             try:
                 resp = self.client.embeddings.create(input=[text], model="text-embedding-3-small")
                 return resp.data[0].embedding
             except Exception:
                 pass

        if self.model is None or self.tokenizer is None:
             return [0.0] * 256

        try:
            import torch
            if not text:
                return [0.0] * 256
                
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1] 
                # Mean pooling
                embedding = hidden_states.mean(dim=1).squeeze().tolist()
                if isinstance(embedding, float):
                    return [embedding]
                return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return [0.0] * 256

    def chat(
        self,
        message: str,
        system_context: Optional[str] = None,
        image: Any = None,
        model_hint: Optional[str] = None,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """
        Generate chat response from Gemma model (local or remote).
        
        Args:
            message: User message text
            system_context: Optional system context (robot status, hardware info, etc.)
            image: Optional image input (PIL Image or numpy array) for VLM
        
        Returns:
            Model response text
        """
        
        # --- PATH 0: FunctionGemma backend ---
        if model_hint == FunctionGemmaChat.MODEL_ID:
            if self._function_chat is None:
                self._function_chat = FunctionGemmaChat(hf_token=self.hf_token)
            return self._function_chat.chat(
                message=message,
                system_context=system_context,
                tools=tools,
                tool_results=tool_results,
            )

        # --- PATH 1: Remote API (OpenAI/vLLM) ---
        if self.client:
            # TODO: Handle image for API if supported (e.g. GPT-4o)
            try:
                model_name = model_hint or self.model_name
                messages = []
                if system_context:
                    messages.append({"role": "system", "content": system_context})
                
                # Add history
                # Ensure history format matches OpenAI (role/content)
                for msg in self.chat_history[-6:]:
                    role = "user" if msg["role"].lower() in ["user"] else "assistant"
                    messages.append({"role": role, "content": msg["content"]})
                    
                messages.append({"role": "user", "content": message})
                
                logger.info(f"Sending request to API: {self.api_base}")
                completion = self.client.chat.completions.create(
                    model=model_name, # vLLM often ignores this or needs it to match loaded model
                    messages=messages,
                    temperature=0.7,
                    max_tokens=256
                )
                
                response = completion.choices[0].message.content
                
                # Update history
                self.chat_history.append({"role": "User", "content": message})
                self.chat_history.append({"role": "Assistant", "content": response})
                if len(self.chat_history) > self.max_history * 2:
                    self.chat_history = self.chat_history[-self.max_history * 2:]
                    
                return response

            except Exception as e:
                logger.error(f"API chat failed: {e}")
                return f"Error from remote API: {str(e)}"
        
        # --- PATH 1.5: Explicit Mock/Debug ---
        if model_hint == "mock":
            return f"Mock response. System Context length: {len(system_context) if system_context else 0}. Message received: {message}"

        # --- PATH 2: Local Transformers ---
        if model_hint and model_hint != self.model_name:

            # Attempt to switch models ONLY if available locally (or downloads explicitly allowed).
            if _model_in_local_hf_cache(model_hint) or _allow_model_downloads():
                self.model_name = model_hint
                self.model = None
                self.tokenizer = None
                self.processor = None
                self.is_vlm = False
            else:
                logger.warning(
                    f"Ignoring model_hint={model_hint} because it is not present in local HF cache "
                    f"({_resolve_hf_hub_dir()}) and downloads are disabled."
                )

        if self.model is None:
            # Pass return_error=True (we need to update load_model signature slightly or just capture it)
            # Actually load_model currently returns bool.
            # Let's verify load_model logic first.
            if not self.load_model():
                # We can't easily get the error string from load_model unless we change it.
                # But we can try the import HERE to see if it fails and why.
                try:
                    import transformers
                    import torch
                    return f"Error: Gemma model failed to load. Check server logs for details."
                except ImportError as e:
                    return f"Error: Gemma model not available. Missing dependency: {e}"
                except Exception as e:
                    return f"Error: Gemma model not available. Unexpected error: {e}"
        
        try:
            import torch
            
            # --- VLM Generation ---
            if self.is_vlm and self.processor:
                # Prepare inputs with Processor
                
                # Combine system context and message
                full_prompt = message
                if system_context:
                    # Gemma 3N chat templates usually handle system instructions differently
                    # For now, prepend simple context
                    full_prompt = f"{system_context}\n\n{message}"
                
                # Add history
                # Note: Multiturn VLM is complex. For now, we just pass the current turn.
                # Or append history text.
                # TODO: Use apply_chat_template if supported by processor
                
                # Handle inputs
                if image:
                    inputs = self.processor(text=full_prompt, images=image, return_tensors="pt")
                else:
                    inputs = self.processor(text=full_prompt, images=None, return_tensors="pt")
                
                # Move pixel_values to correct device/dtype if needed
                # inputs = inputs.to(self.device) # device_map="cpu" handles this usually, but let's be safe if manual
                # actually device_map="cpu" means model is on CPU, inputs should be CPU.
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, 
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7
                    )
                
                response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                
                # Post-process response (remove prompt echo if present)
                # Gemma 3N output usually contains the prompt? check verification script output
                # Usually batch_decode(outputs) includes prompt.
                # We can use skip_prompt=True if processor supports it, or manual slicing.
                if response.startswith(full_prompt):
                    response = response[len(full_prompt):].strip()
                elif "Assistant:" in response:
                     response = response.split("Assistant:")[-1].strip()
                elif "model" in response.lower() and ":" in response: # e.g. "model: answer"
                     pass
                
                # Update history
                self.chat_history.append({"role": "User", "content": message})
                self.chat_history.append({"role": "Assistant", "content": response})
                return response

            # --- CausalLM Generation (Legacy) ---
            
            # Add system context if provided
            if system_context:
                prompt = f"System: {system_context}\n\nUser: {message}\n\nAssistant:"
            else:
                prompt = f"User: {message}\n\nAssistant:"
            
            # Add chat history context
            if self.chat_history:
                history_text = "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in self.chat_history[-6:]  # Last 3 turns
                ])
                prompt = f"{history_text}\n{prompt}"
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response (after "Assistant:")
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            # Update chat history
            self.chat_history.append({"role": "User", "content": message})
            self.chat_history.append({"role": "Assistant", "content": response})
            
            # Trim history
            if len(self.chat_history) > self.max_history * 2:
                self.chat_history = self.chat_history[-self.max_history * 2:]
            
            return response
            
        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def reset_history(self):
        """Clear chat history."""
        self.chat_history = []
        logger.info("Chat history reset")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and status.
        
        Returns:
            Dictionary with model details
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "accelerator": self.accelerator_device,
            "loaded": self.model is not None,
            "history_length": len(self.chat_history) // 2,  # Number of turns
            "has_token": self.hf_token is not None
        }


# Mock implementation for testing without transformers
class MockGemmaChat:
    """Mock Gemma chat for testing without model."""
    
    
    def __init__(self, model_name: str = "mock", device: str = "cpu", accelerator_device: str = None, error_msg: str = None):
        self.model_name = model_name
        self.device = device
        self.error_msg = error_msg
        self.min_history = []
        if accelerator_device:
             logger.info(f" MockGemmaChat using mock accelerator: {accelerator_device}")
        self.chat_history = []
        logger.info(f"Using mock Gemma chat. Error: {error_msg}")
    
    def load_model(self) -> bool:
        return True
    
    def embed(self, text: str) -> List[float]:
        """Generate mock embedding."""
        # Deterministic mock embedding based on hash
        import hashlib
        h = hashlib.md5(text.encode()).digest()
        # Create list of floats from bytes
        return [float(b)/255.0 for b in h] * 16 # Extend to 256 dim

    def chat(self, message: str, system_context: Optional[str] = None, model_hint: Optional[str] = None) -> str:
        """Generate mock response."""
        self.chat_history.append({"role": "User", "content": message})
        
        # Simple pattern matching for mock responses
        msg_lower = message.lower()
        
        if "status" in msg_lower or "how are" in msg_lower:
            response = "I'm a mock Gemma assistant. The robot is operational and ready for commands."
        elif "camera" in msg_lower:
            response = "The OAK-D Lite camera is active and streaming at 640x400 resolution."
        elif "move" in msg_lower or "control" in msg_lower:
            response = "You can control the robot using the joint sliders or arrow buttons on this page."
        elif "help" in msg_lower:
            response = "I can help you with robot status, camera info, and control instructions. What would you like to know?"
        else:
            reason = f" Missing dependency: {self.error_msg}" if self.error_msg else " Install transformers for real Gemma inference."
            response = f"[Mock] You said: '{message}'.{reason}"
        
        self.chat_history.append({"role": "Assistant", "content": response})
        return response
    
    def reset_history(self):
        self.chat_history = []
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": "mock",
            "device": "cpu",
            "loaded": True,
            "history_length": len(self.chat_history) // 2,
            "has_token": False
        }



# Factory function to create appropriate chat instance
def _install_missing_deps():
    """Best-effort install of core deps to avoid mock fallback."""
    # NOTE: Auto-installing dependencies at runtime can violate offline-first and
    # boot-reliability goals (and is undesirable under systemd). Keep this OFF by default.
    if os.environ.get("CONTINUON_AUTO_INSTALL_PY_DEPS", "0").lower() not in ("1", "true", "yes", "on"):
        return False
    required = ["transformers", "accelerate", "timm", "ai-edge-litert", "ai-edge-litert"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return True
    logger.warning(f"Missing dependencies {missing}; attempting pip install")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *missing])
        return True
    except Exception as exc:
        logger.error(f"Auto-install failed: {exc}")
        return False


def create_gemma_chat(use_mock: bool = False, **kwargs) -> Any:
    """
    Create Gemma chat instance.
    
    Args:
        use_mock: Force mock implementation
        **kwargs: Arguments passed to chat constructor
    
    Returns:
        GemmaChat or MockGemmaChat instance
    """
    create_gemma_chat.DEFAULT_MODEL_ID = GemmaChat.DEFAULT_MODEL_ID
    requested_model = kwargs.get("model_name") or GemmaChat.DEFAULT_MODEL_ID
    if requested_model == FunctionGemmaChat.MODEL_ID:
        return FunctionGemmaChat(hf_token=os.environ.get("HUGGINGFACE_TOKEN"))
    if use_mock:
        return MockGemmaChat(**kwargs)
    
    # Check for OpenAI API configuration
    api_base = os.environ.get("LLM_API_BASE")
    api_key = os.environ.get("LLM_API_KEY")
    
    if api_base or api_key:
        logger.info(f"Configuring GemmaChat with remote API: {api_base}")
        # Pass API config to constructor
        kwargs["api_base"] = api_base
        kwargs["api_key"] = api_key
        
        # We still try to import transformers/openai inside the class, 
        # but the class itself handles the logic.
        return GemmaChat(**kwargs)

    try:
        import transformers
        import torch
        return GemmaChat(**kwargs)
    except ImportError as e:
        logger.warning(f"transformers not available: {e}.")
        if _install_missing_deps():
            try:
                import transformers
                import torch
                return GemmaChat(**kwargs)
            except Exception as exc:
                logger.error(f"Deps installed but Gemma init still failed: {exc}")
        return MockGemmaChat(error_msg=str(e), **kwargs)

# Exposed for brain_service fallback logic
create_gemma_chat.DEFAULT_MODEL_ID = GemmaChat.DEFAULT_MODEL_ID


def build_chat_service() -> Optional[Any]:
    """
    Public chat-service builder used by ChatAdapter and eval runners.

    This is the canonical location for chat backend selection so call sites can do:
      from continuonbrain.gemma_chat import build_chat_service

    Policy:
    - Prefer JAX/Flax Gemma when CONTINUON_PREFER_JAX=1 and the backend is available.
    - Otherwise fall back to transformers-based Gemma (local-cache first).
    - On headless targets, transformers fallback is disabled by default to keep boot light.
      Override with CONTINUON_ALLOW_TRANSFORMERS_CHAT=1.
    """
    prefer_jax = os.environ.get("CONTINUON_PREFER_JAX", "1").lower() in ("1", "true", "yes", "on")
    headless = os.environ.get("CONTINUON_HEADLESS", "0").lower() in ("1", "true", "yes", "on")
    allow_transformers = os.environ.get("CONTINUON_ALLOW_TRANSFORMERS_CHAT", "0").lower() in ("1", "true", "yes", "on")
    use_litert = os.environ.get("CONTINUON_USE_LITERT", "1").lower() in ("1", "true", "yes", "on") # Default enabled if avail

    # Best-effort accelerator detection (non-fatal).
    accelerator_device = None
    try:
        hailo_devices = list(Path("/dev").glob("hailo*"))
        if hailo_devices:
            accelerator_device = "hailo8l"
    except Exception:
        accelerator_device = None

    enable_jax_chat = os.environ.get("CONTINUON_ENABLE_JAX_GEMMA_CHAT", "0").lower() in ("1", "true", "yes", "on")
    if prefer_jax and enable_jax_chat:
        try:
            from continuonbrain.gemma_chat_jax import create_gemma_chat_jax

            chat_jax = create_gemma_chat_jax(device="cpu", accelerator_device=accelerator_device)
            if chat_jax:
                logger.info("Using JAX/Flax Gemma chat backend")
                return chat_jax
        except Exception as exc:  # noqa: BLE001
            logger.info(f"JAX/Flax Gemma chat unavailable ({exc}); falling back.")


    # LiteRT Preference
    use_litert = os.environ.get("CONTINUON_USE_LITERT", "1").lower() in ("1", "true", "yes", "on")
    prefer_litert = os.environ.get("CONTINUON_PREFER_LITERT", "0").lower() in ("1", "true", "yes", "on")
    
    # Check for LiteRT availability (lazy check via import attempt or pkg util)
    litert_available = False
    try:
        from continuonbrain.services.chat.litert_chat import HAS_LITERT, LiteRTGemmaChat
        litert_available = HAS_LITERT
    except ImportError:
        pass

    # If LiteRT is PREFERRED, try it BEFORE JAX
    if prefer_litert and litert_available and use_litert:
        try:
             # Check for LiteRT model preference or default
             chat = LiteRTGemmaChat(accelerator_device=accelerator_device)
             logger.info("Using LiteRT (TensorFlow Lite) Gemma chat backend (Preferred)")
             return chat
        except Exception as exc:
             logger.warning(f"LiteRT Chat init failed: {exc}")

    # Fallback to JAX/Flax if enabled
    if prefer_jax and enable_jax_chat:
        try:
            from continuonbrain.gemma_chat_jax import create_gemma_chat_jax

            chat_jax = create_gemma_chat_jax(device="cpu", accelerator_device=accelerator_device)
            if chat_jax:
                logger.info("Using JAX/Flax Gemma chat backend")
                return chat_jax
        except Exception as exc:  # noqa: BLE001
            logger.info(f"JAX/Flax Gemma chat unavailable ({exc}); falling back.")


    if headless and not allow_transformers and not litert_available:
        logger.info("Headless mode: transformers chat disabled and LiteRT unavailable.")
        return None

    # If LiteRT was NOT preferred but is available, try it now (after JAX failed or wasn't preferred)
    if litert_available and use_litert:
         try:
             return LiteRTGemmaChat(accelerator_device=accelerator_device)
         except Exception as exc:
             logger.warning(f"LiteRT Chat init failed: {exc}")

    try:
        return create_gemma_chat(use_mock=False, accelerator_device=accelerator_device)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Transformers Gemma init failed: {exc}")
        return None

