
import json
import time
import os
import re
import shutil
import hashlib
import zipfile
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import deque

logger = logging.getLogger(__name__)

class LearningControllerMixin:
    """
    Controller mixin for handling learning/training API endpoints.
    Assumes access to `self._base_dir()` and `self._json_response()`.
    """

    def _base_dir(self) -> Path:
        # This should be implemented by the main handler or base class.
        # Fallback if not present (though it should be).
        if hasattr(super(), "_base_dir"):
             return super()._base_dir()
        # Default fallback (risky if not aligned with server.py)
        return Path("/opt/continuonos/brain") 

    def _json_response(self, payload: Any, status_code: int = 200) -> bytes:
        # This is expected to be provided by the main handler
        # We define a stub for type checking, but at runtime it uses the main class's method
        raise NotImplementedError("Mixin expects _json_response")

    def _build_cloud_readiness(self) -> Dict[str, Any]:
        """
        Lightweight, file-based readiness report for "cloud TPU v1 training" handoff.
        Intentionally offline-first: no uploads are performed here.
        """
        base_dir = self._base_dir()
        rlds_dir = base_dir / "rlds" / "episodes"
        tfrecord_dir = base_dir / "rlds" / "tfrecord"
        seed_export_dir = base_dir / "model" / "adapters" / "candidate" / "core_model_seed"
        seed_manifest = seed_export_dir / "model_manifest.json"
        ckpt_dir = base_dir / "trainer" / "checkpoints" / "core_model_seed"
        trainer_status = base_dir / "trainer" / "status.json"
        proof = base_dir / "proof_of_learning.json"

        def _count_json(prefix: Optional[str] = None) -> int:
            if not rlds_dir.exists():
                return 0
            if prefix:
                return sum(1 for p in rlds_dir.glob(f"{prefix}*.json") if p.is_file())
            return sum(1 for p in rlds_dir.glob("*.json") if p.is_file())

        episodes_total = _count_json()
        hope_eval = _count_json("hope_eval_")
        facts_eval = _count_json("facts_eval_")

        latest_episode = None
        if rlds_dir.exists():
            eps = sorted([p for p in rlds_dir.glob("*.json") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
            if eps:
                latest_episode = {"path": str(eps[0]), "mtime": eps[0].stat().st_mtime}

        tfrecord_files = []
        if tfrecord_dir.exists():
            tfrecord_files = [p for p in tfrecord_dir.rglob("*") if p.is_file()]

        ckpts = []
        if ckpt_dir.exists():
            ckpts = [p for p in ckpt_dir.rglob("*") if p.is_file()]
        ckpts_sorted = sorted(ckpts, key=lambda p: p.stat().st_mtime, reverse=True)
        latest_ckpt = None
        if ckpts_sorted:
            latest_ckpt = {"path": str(ckpts_sorted[0]), "mtime": ckpts_sorted[0].stat().st_mtime, "size_bytes": ckpts_sorted[0].stat().st_size}

        ready = True
        gates = []

        def gate(name: str, ok: bool, detail: str) -> None:
            nonlocal ready
            gates.append({"name": name, "ok": ok, "detail": detail})
            if not ok:
                ready = False

        gate("episodes_present", episodes_total > 0, f"{episodes_total} episode(s) in {rlds_dir}")
        gate("seed_manifest_present", seed_manifest.exists(), f"manifest at {seed_manifest}" if seed_manifest.exists() else f"missing {seed_manifest}")
        gate("seed_checkpoints_present", len(ckpts) > 0, f"{len(ckpts)} file(s) in {ckpt_dir}" if ckpts else f"missing checkpoints in {ckpt_dir}")

        # Optional-but-helpful evidence signals
        optional = {
            "tfrecord_dir": {"path": str(tfrecord_dir), "exists": tfrecord_dir.exists(), "file_count": len(tfrecord_files)},
            "trainer_status": {"path": str(trainer_status), "exists": trainer_status.exists(), "mtime": trainer_status.stat().st_mtime if trainer_status.exists() else None},
            "proof_of_learning": {"path": str(proof), "exists": proof.exists(), "mtime": proof.stat().st_mtime if proof.exists() else None},
        }

        # High-signal, copyable command suggestions (not executed by the server).
        commands = {
            "zip_episodes": f"cd {base_dir} && zip -r episodes.zip rlds/episodes",
            "tfrecord_convert": f"python -m continuonbrain.jax_models.data.tfrecord_converter --input-dir {base_dir}/rlds/episodes --output-dir {base_dir}/rlds/tfrecord --compress",
            "cloud_tpu_train_template": "python -m continuonbrain.run_trainer --trainer jax --mode tpu --data-path gs://... --output-dir gs://... --config-preset tpu --num-steps 10000",
        }

        return {
            "status": "ok",
            "ready_for_cloud_handoff": ready,
            "gates": gates,
            "rlds": {
                "dir": str(rlds_dir),
                "episodes_total": episodes_total,
                "hope_eval_episodes": hope_eval,
                "facts_eval_episodes": facts_eval,
                "latest_episode": latest_episode,
            },
            "seed": {
                "export_dir": str(seed_export_dir),
                "manifest_path": str(seed_manifest),
                "manifest_exists": seed_manifest.exists(),
                "checkpoint_dir": str(ckpt_dir),
                "checkpoint_file_count": len(ckpts),
                "latest_checkpoint": latest_ckpt,
            },
            "optional": optional,
            "commands": commands,
            "distribution": {
                "options": [
                    {
                        "id": "manual_zip",
                        "label": "Manual zip (download/upload yourself)",
                        "notes": "Build an export zip on-device, copy it to cloud/workstation, then paste the returned bundle URL or path into Install.",
                    },
                    {
                        "id": "signed_edge_bundle",
                        "label": "Signed OTA edge bundle (edge_manifest.json)",
                        "notes": "Preferred for production OTA: signature/checksum gating happens in Continuon AI app + device verifier.",
                    },
                    {
                        "id": "vertex_edge",
                        "label": "Google Vertex AI + Edge distribution (transport)",
                        "notes": "Use Vertex/GCS as the distribution channel; generate a signed URL to a .zip, then install it here (auto-detects bundle type).",
                    },
                ],
                "vertex_templates": {
                    "upload_to_gcs": "gcloud storage cp /opt/continuonos/brain/exports/<EXPORT_ZIP>.zip gs://<BUCKET>/<PREFIX>/<EXPORT_ZIP>.zip",
                    "sign_url_gcloud": "gcloud storage sign-url --duration=1h --private-key-file=<SERVICE_ACCOUNT_KEY.json> gs://<BUCKET>/<PREFIX>/<EXPORT_ZIP>.zip",
                    "sign_url_gsutil_legacy": "gsutil signurl -d 1h <SERVICE_ACCOUNT_KEY.json> gs://<BUCKET>/<PREFIX>/<EXPORT_ZIP>.zip",
                    "vertex_model_registry_hint": "Optional: register the trained bundle metadata in Vertex AI Model Registry for tracking; distribution to devices still uses signed URLs or your OTA pipeline.",
                },
            },
        }

    def _sha256_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _build_cloud_export_zip(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        exports_dir = self._base_dir() / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        name = payload.get("name") or f"cloud_handoff_{ts}.zip"
        # Sanitize: keep it simple and safe.
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
        if not name.endswith(".zip"):
            name += ".zip"
        out_path = (exports_dir / name).resolve()
        if not str(out_path).startswith(str(exports_dir.resolve())):
            raise ValueError("Invalid export name")

        include = payload.get("include") if isinstance(payload.get("include"), dict) else {}
        include_episodes = bool(include.get("episodes", True))
        include_tfrecord = bool(include.get("tfrecord", False))
        include_seed = bool(include.get("seed_export", True))
        include_checkpoints = bool(include.get("checkpoints", True))
        include_status = bool(include.get("trainer_status", True))
        episode_limit = payload.get("episode_limit")
        try:
            episode_limit = int(episode_limit) if episode_limit is not None else None
        except Exception:
            episode_limit = None

        roots = []
        if include_episodes:
            roots.append(("rlds/episodes", self._base_dir() / "rlds" / "episodes"))
        if include_tfrecord:
            roots.append(("rlds/tfrecord", self._base_dir() / "rlds" / "tfrecord"))
        if include_seed:
            roots.append(("model/adapters/candidate/core_model_seed", self._base_dir() / "model" / "adapters" / "candidate" / "core_model_seed"))
        if include_checkpoints:
            roots.append(("trainer/checkpoints/core_model_seed", self._base_dir() / "trainer" / "checkpoints" / "core_model_seed"))
        if include_status:
            roots.append(("trainer/status.json", self._base_dir() / "trainer" / "status.json"))

        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Always include a small metadata record for provenance.
            meta = {
                "created_at_unix_s": int(time.time()),
                "created_at": ts,
                "includes": {
                    "episodes": include_episodes,
                    "tfrecord": include_tfrecord,
                    "seed_export": include_seed,
                    "checkpoints": include_checkpoints,
                    "trainer_status": include_status,
                },
            }
            zf.writestr("handoff_manifest.json", json.dumps(meta, indent=2))

            for arc_root, src in roots:
                if not src.exists():
                    continue
                if src.is_file():
                    zf.write(src, arcname=arc_root)
                    continue

                files = [p for p in src.rglob("*") if p.is_file()]
                if arc_root == "rlds/episodes" and episode_limit and episode_limit > 0:
                    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:episode_limit]
                for f in files:
                    rel = f.relative_to(src)
                    zf.write(f, arcname=str(Path(arc_root) / rel))

        sha256 = self._sha256_file(out_path)
        return {
            "exports_dir": str(exports_dir),
            "zip_name": out_path.name,
            "zip_path": str(out_path),
            "size_bytes": out_path.stat().st_size,
            "sha256": sha256,
            "download_url": f"/api/training/exports/download/{out_path.name}",
        }

    def _download_export_zip(self, name: str) -> bytes:
        exports_dir = (self._base_dir() / "exports").resolve()
        exports_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
        if safe != name or not safe.endswith(".zip"):
            return b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n"
        path = (exports_dir / safe).resolve()
        if not str(path).startswith(str(exports_dir)) or not path.exists() or not path.is_file():
            return b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"
        content = path.read_bytes()
        header = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/zip\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            f"Content-Length: {len(content)}\r\n"
            f"Content-Disposition: attachment; filename=\"{safe}\"\r\n\r\n"
        )
        return header.encode("utf-8") + content

    def _tail_file(self, path: Path, line_count: int) -> List[str]:
        """Return the last ``line_count`` lines from ``path`` safely and efficiently."""

        if line_count <= 0:
            return []

        # Efficiently read the last `line_count` lines from the file
        # Read in binary mode to avoid issues with encoding and newlines
        lines = []
        buffer = b""
        chunk_size = 4096
        try:
            with path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                pos = file_size
                while pos > 0 and len(lines) <= line_count:
                    read_size = min(chunk_size, pos)
                    pos -= read_size
                    f.seek(pos)
                    data = f.read(read_size)
                    buffer = data + buffer
                    # Split lines
                    lines = buffer.splitlines()
            # Return the last `line_count` lines, decoded
            return [line.decode(errors="replace") for line in lines[-line_count:]]
        except Exception:
            # Fallback to original method if any error occurs
            buffer_dq: deque[str] = deque(maxlen=line_count)
            with path.open("r", errors="replace") as handle:
                for line in handle:
                    buffer_dq.append(line)
            return list(buffer_dq)

    def _read_training_metrics(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Read lightweight training metrics for UI visualization (sparklines).
        """
        log_dir = self._base_dir() / "trainer" / "logs"
        limit_raw = (query_params.get("limit", ["120"]) or ["120"])[0]
        try:
            limit = max(10, min(2000, int(limit_raw)))
        except Exception:
            limit = 120

        def read_series(path: Path, *, y_key: str = "loss") -> Dict[str, Any]:
            if not path.exists():
                return {"path": str(path), "exists": False, "points": []}
            try:
                data = json.loads(path.read_text())
                if not isinstance(data, list):
                    return {"path": str(path), "exists": True, "points": []}
                pts = []
                for item in data[-limit:]:
                    if not isinstance(item, dict):
                        continue
                    step = item.get("step")
                    y = item.get(y_key)
                    try:
                        pts.append({"step": int(step), y_key: float(y)})
                    except Exception:
                        continue
                return {"path": str(path), "exists": True, "points": pts, "mtime": path.stat().st_mtime}
            except Exception:
                return {"path": str(path), "exists": True, "points": []}

        return {
            "status": "ok",
            "limit": limit,
            "wavecore": {
                "fast": read_series(log_dir / "wavecore_fast_metrics.json", y_key="loss"),
                "mid": read_series(log_dir / "wavecore_mid_metrics.json", y_key="loss"),
                "slow": read_series(log_dir / "wavecore_slow_metrics.json", y_key="loss"),
            },
            "tool_router": {
                "loss": read_series(log_dir / "tool_router_metrics.json", y_key="loss"),
                "acc": read_series(log_dir / "tool_router_metrics.json", y_key="acc"),
            },
            "tool_router_eval": {
                "top1": read_series(log_dir / "tool_router_eval_metrics.json", y_key="top1"),
                "top5": read_series(log_dir / "tool_router_eval_metrics.json", y_key="top5"),
            },
        }

    def _read_eval_summary(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize recent eval RLDS episodes so the UI can render "intelligence" indicators.
        """
        rlds_dir = self._base_dir() / "rlds" / "episodes"
        limit_raw = (query_params.get("limit", ["6"]) or ["6"])[0]
        try:
            limit = max(1, min(30, int(limit_raw)))
        except Exception:
            limit = 6

        def summarize_prefix(prefix: str) -> Dict[str, Any]:
            if not rlds_dir.exists():
                return {"prefix": prefix, "episodes": [], "latest": None}
            files = sorted(rlds_dir.glob(f"{prefix}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
            episodes = []
            for p in files:
                try:
                    payload = json.loads(p.read_text())
                except Exception:
                    continue
                steps = payload.get("steps", [])
                if not isinstance(steps, list):
                    steps = []
                total = 0
                ok = 0
                fallback = 0
                tiers: Dict[str, int] = {}
                for s in steps:
                    if not isinstance(s, dict):
                        continue
                    total += 1
                    obs = s.get("obs") or s.get("observation") or {}
                    action = s.get("action") or {}
                    tier = (obs.get("tier") if isinstance(obs, dict) else None) or "unknown"
                    tiers[str(tier)] = tiers.get(str(tier), 0) + 1
                    ans = action.get("answer") if isinstance(action, dict) else None
                    used_fb = bool(action.get("used_fallback")) if isinstance(action, dict) else False
                    if used_fb:
                        fallback += 1
                    ans_s = str(ans or "")
                    if ans_s and not ans_s.startswith("[error:"):
                        ok += 1
                episodes.append(
                    {
                        "path": str(p),
                        "mtime": p.stat().st_mtime,
                        "steps": total,
                        "ok_steps": ok,
                        "fallback_steps": fallback,
                        "success_rate": (ok / total) if total else None,
                        "fallback_rate": (fallback / total) if total else None,
                        "tiers": tiers,
                    }
                )
            latest = episodes[0] if episodes else None
            return {"prefix": prefix, "episodes": episodes, "latest": latest}

        return {
            "status": "ok",
            "rlds_dir": str(rlds_dir),
            "limit": limit,
            "hope_eval": summarize_prefix("hope_eval"),
            "facts_eval": summarize_prefix("facts_eval"),
            "compare_eval": summarize_prefix("compare_eval"),
            "wiki_learn": summarize_prefix("wiki_learn"),
        }

    def _read_data_quality(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Quick RLDS JSON "learnability" stats for UI.
        """
        rlds_dir = self._base_dir() / "rlds" / "episodes"
        limit_raw = (query_params.get("limit", ["30"]) or ["30"])[0]
        step_cap_raw = (query_params.get("step_cap", ["2500"]) or ["2500"])[0]
        try:
            limit = max(1, min(200, int(limit_raw)))
        except Exception:
            limit = 30
        try:
            step_cap = max(200, min(20000, int(step_cap_raw)))
        except Exception:
            step_cap = 2500

        if not rlds_dir.exists():
            return {"status": "ok", "rlds_dir": str(rlds_dir), "episodes_scanned": 0, "steps_scanned": 0}

        files = sorted(rlds_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]

        steps_scanned = 0
        action_present = 0
        action_nonzero = 0
        action_abs_sum = 0.0
        action_abs_sum_sq = 0.0
        action_len_total = 0

        obs_present = 0
        obs_numeric_fields = 0
        obs_numeric_scalars = 0

        obs_key_counts: Dict[str, int] = {}
        episode_kind_counts: Dict[str, int] = {}

        def _flatten_numeric(x: Any) -> List[float]:
            out: List[float] = []
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                out.append(float(x))
                return out
            if isinstance(x, (list, tuple)):
                for v in x:
                    out.extend(_flatten_numeric(v))
            if isinstance(x, dict):
                for v in x.values():
                    out.extend(_flatten_numeric(v))
            return out

        def _action_vec(step: Dict[str, Any]) -> Optional[List[float]]:
            action = step.get("action")
            if not isinstance(action, dict):
                return None
            cmd = action.get("command")
            if isinstance(cmd, (list, tuple)) and cmd:
                try:
                    return [float(v) for v in cmd]
                except Exception:
                    return None
            return None

        # Simplified scanning loop for brevity (omitted full stats for now if not critical, but included core)
        # Actually, let's include the core logic from routes.py to be safe
        for p in files:
            if steps_scanned >= step_cap:
                break
            try:
                payload = json.loads(p.read_text())
            except Exception:
                continue
            steps = payload.get("steps", [])
            if not isinstance(steps, list):
                continue
            
            # Episode kind heuristic
            kind = "unknown"
            base = p.name
            if base.startswith("hope_eval_") or base.startswith("hope_eval_followup_"):
                kind = "hope_eval"
            elif base.startswith("facts_eval_"):
                kind = "facts_eval"
            elif base.startswith("compare_eval_"):
                kind = "compare_eval"
            elif base.startswith("test_"):
                kind = "test"
            episode_kind_counts[kind] = episode_kind_counts.get(kind, 0) + 1

            for s in steps:
                if steps_scanned >= step_cap:
                    break
                if not isinstance(s, dict):
                    continue
                steps_scanned += 1

                vec = _action_vec(s)
                if vec is not None:
                    action_present += 1
                    action_len_total += len(vec)
                    abs_sum = sum(abs(v) for v in vec)
                    action_abs_sum += abs_sum
                    action_abs_sum_sq += abs_sum * abs_sum
                    if abs_sum > 1e-9:
                        action_nonzero += 1

                obs = s.get("observation")
                if isinstance(obs, dict):
                    obs_present += 1
                    for k in obs.keys():
                        obs_key_counts[k] = obs_key_counts.get(k, 0) + 1
                    nums = _flatten_numeric(obs)
                    if nums:
                        obs_numeric_fields += 1
                        obs_numeric_scalars += len(nums)

        action_present_rate = (action_present / steps_scanned) if steps_scanned else None
        action_nonzero_rate = (action_nonzero / action_present) if action_present else None
        action_mean_abs_sum = (action_abs_sum / action_present) if action_present else None
        action_std_abs_sum = None
        if action_present and action_mean_abs_sum is not None:
            mean = action_mean_abs_sum
            var = (action_abs_sum_sq / action_present) - (mean * mean)
            if var < 0:
                var = 0.0
            action_std_abs_sum = var ** 0.5

        obs_present_rate = (obs_present / steps_scanned) if steps_scanned else None
        obs_numeric_rate = (obs_numeric_fields / obs_present) if obs_present else None
        obs_avg_numeric_scalars = (obs_numeric_scalars / obs_numeric_fields) if obs_numeric_fields else None

        top_obs_keys = sorted(obs_key_counts.items(), key=lambda kv: kv[1], reverse=True)[:12]

        warnings = []
        if steps_scanned and (action_present == 0):
            warnings.append("No action.command vectors found.")
        if action_nonzero_rate is not None and action_nonzero_rate < 0.1:
            warnings.append("Most action vectors are near-zero.")
        if obs_present == 0:
            warnings.append("No observation dicts found.")
        if obs_numeric_rate is not None and obs_numeric_rate < 0.2:
            warnings.append("Most observations contain little/no numeric content.")

        return {
            "status": "ok",
            "rlds_dir": str(rlds_dir),
            "episodes_scanned": len(files),
            "steps_scanned": steps_scanned,
            "episode_kinds": episode_kind_counts,
            "action": {
                "present_rate": action_present_rate,
                "nonzero_rate": action_nonzero_rate,
                "avg_len": (action_len_total / action_present) if action_present else None,
                "mean_abs_sum": action_mean_abs_sum,
                "std_abs_sum": action_std_abs_sum,
            },
            "observation": {
                "present_rate": obs_present_rate,
                "numeric_rate": obs_numeric_rate,
                "avg_numeric_scalars": obs_avg_numeric_scalars,
                "top_keys": [{"key": k, "count": c} for k, c in top_obs_keys],
            },
            "warnings": warnings,
        }

    def _read_tool_dataset_summary(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize imported tool-use RLDS episodes.
        """
        base_dir = self._base_dir() / "rlds" / "episodes"
        limit_raw = (query_params.get("limit", ["2000"]) or ["2000"])[0]
        try:
            limit = max(50, min(200000, int(limit_raw)))
        except Exception:
            limit = 2000

        subdirs = []
        if base_dir.exists():
            for p in sorted(base_dir.glob("toolchat_hf*")):
                if p.is_dir():
                    subdirs.append(p)

        def summarize_dir(d: Path) -> Dict[str, Any]:
            files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
            episodes = 0
            steps_total = 0
            tool_call_steps = 0
            dataset_ids: Dict[str, int] = {}
            tool_names: Dict[str, int] = {}

            for fp in files:
                try:
                    payload = json.loads(fp.read_text())
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                episodes += 1
                meta = payload.get("metadata") or {}
                dsid = meta.get("dataset_id")
                if isinstance(dsid, str) and dsid:
                    dataset_ids[dsid] = dataset_ids.get(dsid, 0) + 1
                steps = payload.get("steps", [])
                if not isinstance(steps, list):
                    continue
                steps_total += len(steps)
                for s in steps:
                    if not isinstance(s, dict):
                        continue
                    action = s.get("action") or {}
                    if not isinstance(action, dict):
                        continue
                    if action.get("type") == "tool_call":
                        tool_call_steps += 1
                        nm = action.get("name")
                        if isinstance(nm, str) and nm:
                            tool_names[nm] = tool_names.get(nm, 0) + 1

            top_tools = sorted(tool_names.items(), key=lambda kv: kv[1], reverse=True)[:20]
            tool_call_rate = (tool_call_steps / steps_total) if steps_total else None
            avg_steps = (steps_total / episodes) if episodes else None
            top_dataset_ids = sorted(dataset_ids.items(), key=lambda kv: kv[1], reverse=True)[:8]

            return {
                "dir": str(d),
                "episodes": episodes,
                "steps_total": steps_total,
                "avg_steps_per_episode": avg_steps,
                "tool_call_steps": tool_call_steps,
                "tool_call_rate": tool_call_rate,
                "top_tools": [{"name": n, "count": c} for n, c in top_tools],
                "dataset_ids": [{"id": i, "episodes": c} for i, c in top_dataset_ids],
            }

        return {
            "status": "ok",
            "base_dir": str(base_dir),
            "limit": limit,
            "sources": [summarize_dir(d) for d in subdirs],
        }

    def _install_model_bundle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Install a returned artifact from "distribution" (manual URL/path).
        """
        kind = payload.get("kind") or "jax_seed_manifest"
        source_url = payload.get("source_url")
        source_path = payload.get("source_path")
        if not source_url and not source_path:
            raise ValueError("Provide source_url or source_path")

        incoming_root = self._base_dir() / "model" / "_incoming"
        incoming_root.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        staging = incoming_root / f"incoming_{ts}"
        staging.mkdir(parents=True, exist_ok=True)

        # Fetch/copy into staging set local_in
        local_in = None
        if source_path:
            p = Path(str(source_path)).expanduser()
            if not p.is_absolute():
                raise ValueError("source_path must be absolute")
            if not p.exists():
                raise FileNotFoundError(str(p))
            if p.is_dir():
                local_in = staging / "payload_dir"
                shutil.copytree(p, local_in)
            else:
                local_in = staging / p.name
                shutil.copy2(p, local_in)
        else:
            # URL download (best-effort; keep minimal deps)
            import urllib.request

            dest = staging / "download.bin"
            req = urllib.request.Request(str(source_url), headers={"User-Agent": "continuonbrain-ui/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                with dest.open("wb") as f:
                    shutil.copyfileobj(resp, f)
            local_in = dest

        extracted = staging / "extracted"
        extracted.mkdir(parents=True, exist_ok=True)

        if local_in.is_dir():
            extracted = local_in
        else:
            if local_in.suffix.lower() != ".zip":
                raise ValueError("Expected a .zip file (or a directory) for install")
            with zipfile.ZipFile(local_in, "r") as zf:
                zf.extractall(extracted)

        if kind in {"vertex_edge", "auto"}:
            if list(extracted.rglob("edge_manifest.json")):
                kind = "edge_bundle"
            elif list(extracted.rglob("model_manifest.json")):
                kind = "jax_seed_manifest"
            else:
                raise ValueError("Unable to auto-detect bundle type (expected edge_manifest.json or model_manifest.json)")

        if kind == "jax_seed_manifest":
            manifest_candidates = list(extracted.rglob("model_manifest.json"))
            if not manifest_candidates:
                raise ValueError("model_manifest.json not found in bundle")
            manifest_path = manifest_candidates[0]
            
            target_dir = self._base_dir() / "model" / "adapters" / "candidate" / "core_model_seed"
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            backup_dir = None
            if target_dir.exists():
                backup_dir = target_dir.parent / f"{target_dir.name}_backup_{ts}"
                shutil.move(str(target_dir), str(backup_dir))
            target_dir.mkdir(parents=True, exist_ok=True)

            src_root = manifest_path.parent
            for p in src_root.rglob("*"):
                if p.is_dir():
                    continue
                rel = p.relative_to(src_root)
                dest = target_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)

            return {
                "kind": kind,
                "installed_to": str(target_dir),
                "backup_dir": str(backup_dir) if backup_dir else None,
                "manifest_path": str(target_dir / "model_manifest.json"),
                "notes": "Installed as candidate JAX seed.",
            }

        if kind == "edge_bundle":
            edge_candidates = list(extracted.rglob("edge_manifest.json"))
            if not edge_candidates:
                raise ValueError("edge_manifest.json not found in bundle")
            edge_path = edge_candidates[0]
            edge = json.loads(edge_path.read_text())
            version = str(edge.get("version") or ts)
            version_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", version)
            bundles_dir = self._base_dir() / "model" / "bundles"
            bundles_dir.mkdir(parents=True, exist_ok=True)
            bundle_dir = bundles_dir / version_safe
            if bundle_dir.exists():
                bundle_dir = bundles_dir / f"{version_safe}_{ts}"
            shutil.copytree(edge_path.parent, bundle_dir)
            return {
                "kind": kind,
                "installed_to": str(bundle_dir),
                "edge_manifest": str(bundle_dir / "edge_manifest.json"),
                "notes": "Bundle staged.",
            }

        raise ValueError(f"Unknown install kind: {kind}")
