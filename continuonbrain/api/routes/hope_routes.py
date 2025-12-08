"""
HOPE Brain Monitoring API Routes

Provides endpoints for real-time HOPE brain metrics, memory inspection,
stability monitoring, and performance benchmarking.
"""

import json
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

# Global reference to HOPE brain (will be set by server)
_hope_brain = None
_background_learner = None  # Reference to background learner for progress tracking
_metrics_history = []
_max_history = 1000  # Keep last 1000 steps


def set_hope_brain(brain):
    """Set the global HOPE brain instance."""
    global _hope_brain
    _hope_brain = brain


def set_background_learner(learner):
    """Set the global background learner instance."""
    global _background_learner
    _background_learner = learner


def get_current_metrics() -> Dict[str, Any]:
    """Get current HOPE brain metrics."""
    if _hope_brain is None:
        return {"error": "HOPE brain not initialized"}
    
    try:
        state = _hope_brain.get_state()
        
        # Import stability functions
        from continuonbrain.hope_impl.stability import lyapunov_total, lyapunov_fast_state, lyapunov_memory, lyapunov_params
        import torch
        
        # Compute Lyapunov energies
        V_total = lyapunov_total(state).item()
        V_fast = lyapunov_fast_state(state.fast_state).item()
        V_mem = lyapunov_memory(state.cms).item()
        V_params = lyapunov_params(state.params).item()
        
        # State norms
        s_norm = torch.norm(state.fast_state.s).item()
        w_norm = torch.norm(state.fast_state.w).item()
        p_norm = torch.norm(state.fast_state.p).item()
        
        # CMS utilization (fraction of max norm)
        cms_util = []
        for level in state.cms.levels:
            M_norm = torch.norm(level.M).item()
            # Estimate max based on size
            max_norm = (level.M.shape[0] * level.M.shape[1]) ** 0.5
            cms_util.append(M_norm / max_norm if max_norm > 0 else 0.0)
        
        # Get stability metrics
        stability_metrics = _hope_brain.stability_monitor.get_metrics()
        gradient_norm = stability_metrics.get("gradient_norm")
        gradient_clip = getattr(_hope_brain.config, "gradient_clip", None)
        current_gradient_norm = gradient_norm if gradient_norm is not None else 0.0

        metrics = {
            "lyapunov": {
                "total": V_total,
                "fast": V_fast,
                "memory": V_mem,
                "params": V_params,
            },
            "state_norms": {
                "s": s_norm,
                "w": w_norm,
                "p": p_norm,
            },
            "cms_utilization": cms_util,
            "learning_rate": state.params.eta,
            "steps": stability_metrics.get("steps", 0),
            "gradient_norm": current_gradient_norm,
            "gradient_spike": bool(
                gradient_clip is not None
                and gradient_norm is not None
                and gradient_norm > gradient_clip
            ),
            "timestamp": time.time(),
        }
        
        # Add learning progress if background learner is available
        if _background_learner is not None:
            try:
                learner_status = _background_learner.get_status()
                metrics["learning_progress"] = {
                    "total_steps": learner_status.get("total_steps", 0),
                    "total_episodes": learner_status.get("total_episodes", 0),
                    "learning_updates": learner_status.get("learning_updates", 0),
                    "avg_parameter_change": learner_status.get("avg_parameter_change", 0.0),
                    "recent_parameter_change": learner_status.get("recent_parameter_change", 0.0),
                    "avg_episode_reward": learner_status.get("avg_episode_reward", 0.0),
                    "recent_episode_reward": learner_status.get("recent_episode_reward", 0.0),
                    "is_learning": learner_status.get("running", False) and not learner_status.get("paused", True),
                }
            except Exception:
                # Don't fail if learner status unavailable
                pass
        
        # Add to history
        _metrics_history.append(metrics)
        if len(_metrics_history) > _max_history:
            _metrics_history.pop(0)
        
        return metrics
        
    except Exception as e:
        return {"error": str(e)}


def get_metrics_history(window: int = 100, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    """Get historical metrics for plotting."""
    if not _metrics_history:
        return {"data": [], "count": 0}
    
    # Get last 'window' entries
    history = _metrics_history[-window:]
    
    # Filter metrics if specified
    if metrics:
        filtered = []
        for entry in history:
            filtered_entry = {"timestamp": entry["timestamp"]}
            for metric in metrics:
                if "." in metric:
                    # Nested metric like "lyapunov.total"
                    parts = metric.split(".")
                    value = entry
                    for part in parts:
                        value = value.get(part, None)
                        if value is None:
                            break
                    if value is not None:
                        filtered_entry[metric] = value
                else:
                    if metric in entry:
                        filtered_entry[metric] = entry[metric]
            filtered.append(filtered_entry)
        history = filtered
    
    return {
        "data": history,
        "count": len(history),
    }


def get_cms_level_stats(level_id: int) -> Dict[str, Any]:
    """Get detailed statistics for a specific CMS level."""
    if _hope_brain is None:
        return {"error": "HOPE brain not initialized"}
    
    try:
        state = _hope_brain.get_state()
        
        if level_id < 0 or level_id >= state.cms.num_levels:
            return {"error": f"Invalid level_id: {level_id}"}
        
        level = state.cms.levels[level_id]
        import torch
        
        # Memory matrix statistics
        M_mean = level.M.mean().item()
        M_std = level.M.std().item()
        M_norm = torch.norm(level.M).item()
        M_min = level.M.min().item()
        M_max = level.M.max().item()
        
        # Key matrix statistics
        K_mean = level.K.mean().item()
        K_std = level.K.std().item()
        K_norm = torch.norm(level.K).item()
        
        # Downsample matrices for visualization (if too large)
        M_shape = level.M.shape
        K_shape = level.K.shape

        # Only send full matrices if small enough
        M_data = None
        K_data = None
        if M_shape[0] * M_shape[1] < 10000:  # < 10k elements
            # Detach to avoid disrupting gradient tracking while monitoring
            M_data = level.M.detach().cpu().numpy().tolist()
        if K_shape[0] * K_shape[1] < 10000:
            K_data = level.K.detach().cpu().numpy().tolist()
        
        return {
            "level_id": level_id,
            "decay": level.decay,
            "shape": {"M": list(M_shape), "K": list(K_shape)},
            "M_stats": {
                "mean": M_mean,
                "std": M_std,
                "norm": M_norm,
                "min": M_min,
                "max": M_max,
            },
            "K_stats": {
                "mean": K_mean,
                "std": K_std,
                "norm": K_norm,
            },
            "M_data": M_data,
            "K_data": K_data,
        }
        
    except Exception as e:
        return {"error": str(e)}


def get_stability_analysis() -> Dict[str, Any]:
    """Get comprehensive stability analysis."""
    if _hope_brain is None:
        return {"error": "HOPE brain not initialized"}
    
    try:
        metrics = _hope_brain.stability_monitor.get_metrics()
        state = _hope_brain.get_state()
        config = _hope_brain.config

        from continuonbrain.hope_impl.stability import lyapunov_total
        import torch
        
        # Check for numerical issues across fast state, CMS levels, and parameter tensors
        has_nan = False
        has_inf = False
        component_checks = []

        def record_component(name: str, tensor) -> None:
            nonlocal has_nan, has_inf
            component_has_nan = torch.isnan(tensor).any().item()
            component_has_inf = torch.isinf(tensor).any().item()

            has_nan = has_nan or component_has_nan
            has_inf = has_inf or component_has_inf

            component_checks.append(
                {
                    "component": name,
                    "has_nan": bool(component_has_nan),
                    "has_inf": bool(component_has_inf),
                }
            )

        # Fast-state components
        record_component("fast_state.s", state.fast_state.s)
        record_component("fast_state.w", state.fast_state.w)
        record_component("fast_state.p", state.fast_state.p)

        # CMS memory levels (both memory and key matrices)
        for level_idx, level in enumerate(state.cms.levels):
            record_component(f"cms.levels[{level_idx}].M", level.M)
            record_component(f"cms.levels[{level_idx}].K", level.K)

        # Adaptable parameter tensors
        for name, tensor in state.params.theta.items():
            record_component(f"params.theta[{name}]", tensor)

        # Stability flags
        is_stable = _hope_brain.stability_monitor.is_stable()
        gradient_clip = getattr(_hope_brain.config, "gradient_clip", None)
        gradient_norm = metrics.get("gradient_norm")
        gradient_spike = gradient_clip is not None and metrics.get("gradient_norm", 0.0) > gradient_clip
        lyapunov_threshold = getattr(config, "lyapunov_threshold", None)
        dissipation_floor = getattr(config, "dissipation_floor", 0.0)
        dissipation_rate = metrics.get("dissipation_rate", 0.0)
        dissipation_breach = dissipation_rate < dissipation_floor if dissipation_rate is not None else False

        return {
            "is_stable": is_stable,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "lyapunov_current": metrics.get("lyapunov_current", 0.0),
            "lyapunov_mean": metrics.get("lyapunov_mean", 0.0),
            "lyapunov_threshold": lyapunov_threshold,
            "dissipation_rate": dissipation_rate,
            "dissipation_floor": dissipation_floor,
            "dissipation_breach": dissipation_breach,
            "state_norm": metrics.get("state_norm", 0.0),
            "gradient_norm": metrics.get("gradient_norm", 0.0),
            "gradient_spike": gradient_spike,
            "steps": metrics.get("steps", 0),
            "component_checks": component_checks,
        }
        
    except Exception as e:
        return {"error": str(e)}


def get_hope_config() -> Dict[str, Any]:
    """Get current HOPE configuration."""
    if _hope_brain is None:
        return {"error": "HOPE brain not initialized"}
    
    try:
        config = _hope_brain.config
        
        return {
            "d_s": config.d_s,
            "d_w": config.d_w,
            "d_p": config.d_p,
            "d_e": config.d_e,
            "d_k": config.d_k,
            "d_c": config.d_c,
            "num_levels": config.num_levels,
            "cms_sizes": config.cms_sizes,
            "cms_dims": config.cms_dims,
            "cms_decays": config.cms_decays,
            "use_quantization": config.use_quantization,
            "learning_rate": config.learning_rate,
            "eta_init": config.eta_init,
            "gradient_clip": config.gradient_clip,
            "lyapunov_threshold": config.lyapunov_threshold,
            "dissipation_floor": config.dissipation_floor,
            "use_layer_norm": config.use_layer_norm,
            "lyapunov_weight": config.lyapunov_weight,
        }
        
    except Exception as e:
        return {"error": str(e)}


def handle_hope_stream(handler):
    """Handle SSE stream for real-time metrics."""
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()

    last_step = -1
    
    try:
        while True:
            # Get metrics
            data = get_current_metrics()
            
            # Send event
            message = f"data: {json.dumps(data)}\n\n"
            handler.wfile.write(message.encode("utf-8"))
            handler.wfile.flush()
            
            # Sleep (10Hz max, or sync with steps if possible)
            time.sleep(0.1)
            
    except (ConnectionResetError, BrokenPipeError):
        # Client disconnected
        pass
    except Exception as e:
        print(f"Stream error: {e}")


def handle_hope_post(handler, body=None):
    """Handle HOPE API POST requests."""
    try:
        # If body was not passed, try to read it (though it might be consumed if server.py read it)
        if body is None:
            content_len = int(handler.headers.get('Content-Length', 0))
            body = handler.rfile.read(content_len).decode('utf-8')
            
        data = json.loads(body) if body else {}
        path = handler.path

        if _hope_brain is None:
            handler.send_json({"error": "HOPE brain not initialized"}, status=503)
            return

        if path == "/api/hope/reset":
            _hope_brain.reset()
            # Also reset stability monitor
            if hasattr(_hope_brain, "stability_monitor"):
                _hope_brain.stability_monitor.reset()
            handler.send_json({"success": True, "message": "Brain reset"})

        elif path == "/api/hope/compact":
            result = _hope_brain.compact_memory()
            handler.send_json({"success": True, "result": result})

        elif path == "/api/hope/learning_rate":
            lr = float(data.get("learning_rate", 0.001))
            # Set global learning rate
            _hope_brain.config.learning_rate = lr
            # If supported, update current state params
            state = _hope_brain.get_state()
            if state and hasattr(state.params, "eta"):
                state.params.eta = lr
            handler.send_json({"success": True, "learning_rate": lr})

        else:
            handler.send_json({"error": "Unknown endpoint"}, status=404)

    except Exception as e:
        handler.send_json({"error": str(e)}, status=500)


def handle_hope_request(handler, body=None):
    """Handle HOPE API requests."""
    path = handler.path
    
    # Check method via the handler object command
    if handler.command == "POST":
        handle_hope_post(handler, body)
        return

    # Special handling for stream (GET)
    if path == "/api/hope/stream":
        handle_hope_stream(handler)
        return
    
    try:
        if path == "/api/hope/metrics":
            # Get current metrics
            data = get_current_metrics()
            handler.send_json(data)
            
        elif path.startswith("/api/hope/history"):
            # Parse query params
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(path)
            params = parse_qs(parsed.query)
            
            window = int(params.get("window", [100])[0])
            metrics = params.get("metrics", [None])[0]
            if metrics:
                metrics = metrics.split(",")
            
            data = get_metrics_history(window=window, metrics=metrics)
            handler.send_json(data)
            
        elif path.startswith("/api/hope/cms/level/"):
            # Extract level_id
            level_id = int(path.split("/")[-1])
            data = get_cms_level_stats(level_id)
            handler.send_json(data)
            
        elif path == "/api/hope/stability":
            data = get_stability_analysis()
            handler.send_json(data)
            
        elif path == "/api/hope/config":
            data = get_hope_config()
            handler.send_json(data)
            
        else:
            handler.send_json({"error": "Unknown endpoint"}, status=404)
            
    except Exception as e:
        handler.send_json({"error": str(e)}, status=500)
