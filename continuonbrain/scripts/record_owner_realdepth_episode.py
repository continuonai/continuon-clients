"""
Record an RLDS episode using a local camera (RGB + optional RealSense depth).

This is intended to create *local-only* training episodes containing:
- camera frames (saved to disk, referenced by URI)
- HOPE-style dialog steps that encode owner identity + curiosity / guidance preference
- a compact numeric observation vector (`observation.command`) so the current
  WaveCore trainer can consume it today.

Usage (Windows host python or WSL with camera passthrough configured):
  python -m continuonbrain.scripts.record_owner_realdepth_episode \\
    --out-dir continuonbrain/rlds/episodes \\
    --episode-id owner_craig_20251216_120000 \\
    --owner-display-name \"Craig Michael Merry\" \\
    --owner-preferred-name \"Craig\" \\
    --steps 12 --interval-s 0.5 --camera-index 0 --source auto

Note: do NOT commit real captures into git. Keep them in /opt/continuonos/brain/rlds/episodes on-device.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from continuonbrain.recording.camera_episode_recorder import CameraEpisodeRecorder, CaptureConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Record a local owner-identity RLDS episode (RGB + optional depth).")
    p.add_argument("--out-dir", type=Path, default=Path("/opt/continuonos/brain/rlds/episodes"))
    p.add_argument("--episode-id", type=str, required=True)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--interval-s", type=float, default=0.5)
    p.add_argument("--obs-dim", type=int, default=128)
    p.add_argument("--action-dim", type=int, default=32)
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--source", type=str, default="auto", choices=["auto", "opencv", "realsense"])
    p.add_argument("--teacher-url", type=str, default=None, help="Optional HTTP teacher endpoint to enrich episodes.")
    p.add_argument("--teacher-timeout-s", type=float, default=5.0)
    p.add_argument("--teacher-openai-base-url", type=str, default=None, help="OpenAI-compatible base URL (e.g. http://localhost:8000).")
    p.add_argument("--teacher-openai-embed-base-url", type=str, default=None, help="Optional separate base URL for embeddings (if served on a different port).")
    p.add_argument("--teacher-openai-chat-base-url", type=str, default=None, help="Optional separate base URL for chat (if served on a different port).")
    p.add_argument("--teacher-openai-api-key", type=str, default=None, help="Optional API key for OpenAI-compatible server.")
    p.add_argument("--teacher-openai-embed-model", type=str, default=None, help="Embeddings model name for /v1/embeddings.")
    p.add_argument("--teacher-openai-chat-model", type=str, default=None, help="Chat model name for /v1/chat/completions (optional).")

    p.add_argument("--owner-display-name", type=str, default=None)
    p.add_argument("--owner-preferred-name", type=str, default=None)
    p.add_argument("--owner-id", type=str, default=None)
    p.add_argument("--owner-role", action="append", default=None, help="Repeatable. Example: --owner-role creator --owner-role owner")
    p.add_argument("--instruction", type=str, default="Learn the owner identity and ask for guidance politely.")

    args = p.parse_args()
    if not str(args.episode_id).strip():
        raise SystemExit("--episode-id must be a non-empty string")

    cfg = CaptureConfig(
        out_dir=args.out_dir,
        episode_id=args.episode_id,
        num_steps=args.steps,
        step_interval_s=args.interval_s,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        source=args.source,
    )

    recorder = CameraEpisodeRecorder(cfg)
    teacher = None
    if args.teacher_openai_base_url:
        try:
            from continuonbrain.synthetic.teachers import OpenAITeacher

            teacher = OpenAITeacher(
                base_url=args.teacher_openai_base_url,
                embed_base_url=args.teacher_openai_embed_base_url,
                chat_base_url=args.teacher_openai_chat_base_url,
                api_key=args.teacher_openai_api_key,
                embed_model=args.teacher_openai_embed_model,
                chat_model=args.teacher_openai_chat_model,
                timeout_s=args.teacher_timeout_s,
            )
        except Exception:
            teacher = None
    elif args.teacher_url:
        try:
            from continuonbrain.synthetic.teachers import HttpTeacher

            teacher = HttpTeacher(args.teacher_url, timeout_s=args.teacher_timeout_s)
        except Exception:
            teacher = None
    path = recorder.record_episode(
        owner_display_name=args.owner_display_name,
        owner_preferred_name=args.owner_preferred_name,
        owner_id=args.owner_id,
        owner_roles=args.owner_role,
        instruction=args.instruction,
        privacy_local_only=True,
        teacher=teacher,
    )
    print(f"✅ Wrote episode: {path}")


if __name__ == "__main__":
    main()


