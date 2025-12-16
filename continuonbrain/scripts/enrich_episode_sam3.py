"""
Offline SAM3 segmentation enrichment for an existing RLDS episode.

This script reads an RLDS JSON episode that already references RGB frames in
`steps[*].observation.media.rgb.uri`, then writes back a v1.1 optional
`steps[*].observation.segmentation` block containing:
  - model (string)
  - prompt (string)
  - masks[] with PNG URIs saved alongside the episode blobs
  - boxes_xyxy[] (xyxy pixel coords)

Design goals:
- Dependency-light by default: if torch/transformers/sam3 aren't installed, we
  exit with a helpful message rather than breaking imports elsewhere.
- Runs offline/async from capture so it doesn't starve WaveCore training.

Model reference: `facebook/sam3` ([Hugging Face model card](https://huggingface.co/facebook/sam3)).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _try_import_sam3() -> Tuple[Optional[object], Optional[object], Optional[str]]:
    """
    Attempt to import SAM3 via transformers.
    Returns (Sam3Model, Sam3Processor, error_message).
    """
    try:
        import torch  # noqa: F401
        from transformers import Sam3Model, Sam3Processor  # type: ignore

        return Sam3Model, Sam3Processor, None
    except Exception as exc:
        return None, None, str(exc)


def _ensure_png_write(img_mask: "Any", out_path: Path) -> None:
    try:
        import numpy as np

        arr = img_mask
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
        arr = (arr > 0).astype("uint8") * 255

        try:
            import cv2  # type: ignore

            cv2.imwrite(str(out_path), arr)
            return
        except Exception:
            pass

        # PIL fallback
        from PIL import Image  # type: ignore

        Image.fromarray(arr).save(str(out_path))
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to write mask PNG: {exc}") from exc


def _segment_one_image(
    *,
    model: "Any",
    processor: "Any",
    image_path: Path,
    prompt: str,
    threshold: float,
    device: str,
) -> Tuple[List["Any"], List[List[float]], List[float]]:
    """
    Returns (masks, boxes_xyxy, scores).
    """
    from PIL import Image  # type: ignore
    import torch  # type: ignore

    image = Image.open(str(image_path)).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=float(threshold),
        mask_threshold=float(threshold),
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    masks = results.get("masks")
    boxes = results.get("boxes")
    scores = results.get("scores")
    masks_list = [masks[i] for i in range(masks.shape[0])] if masks is not None else []
    boxes_list = boxes.cpu().tolist() if boxes is not None else []
    scores_list = scores.cpu().tolist() if scores is not None else []
    return masks_list, boxes_list, scores_list


def main() -> None:
    p = argparse.ArgumentParser(description="Offline-enrich an RLDS episode with SAM3 segmentation artifacts.")
    p.add_argument("--episode", type=Path, required=True, help="Path to RLDS episode JSON file")
    p.add_argument("--prompt", type=str, required=True, help="Text prompt for SAM3 segmentation (e.g., 'person', 'hand', 'mug')")
    p.add_argument("--model", type=str, default="facebook/sam3", help="SAM3 model id or local path")
    p.add_argument("--threshold", type=float, default=0.5, help="Mask/instance threshold")
    p.add_argument("--max-steps", type=int, default=0, help="If >0, only process the first N steps")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device")
    args = p.parse_args()

    Sam3Model, Sam3Processor, err = _try_import_sam3()
    if Sam3Model is None or Sam3Processor is None:
        raise SystemExit(
            "SAM3 deps not installed. Install torch + transformers with SAM3 support, and ensure you can access the model.\n"
            f"Import error: {err}\n"
            "Model card: https://huggingface.co/facebook/sam3"
        )

    import torch  # type: ignore

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = Sam3Model.from_pretrained(args.model).to(device)
    processor = Sam3Processor.from_pretrained(args.model)

    episode_path: Path = args.episode
    episode = _load_json(episode_path)

    steps = episode.get("steps", [])
    if not isinstance(steps, list) or not steps:
        raise SystemExit("Episode has no steps")

    blobs_dir = episode_path.parent / f"{episode_path.stem}_blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)

    limit = int(args.max_steps)
    if limit > 0:
        steps = steps[:limit]

    for idx, step in enumerate(steps):
        obs = step.get("observation", {}) if isinstance(step, dict) else {}
        media = obs.get("media", {}) if isinstance(obs, dict) else {}
        rgb = media.get("rgb", {}) if isinstance(media, dict) else {}
        rgb_uri = rgb.get("uri") if isinstance(rgb, dict) else None
        frame_id = rgb.get("frame_id") if isinstance(rgb, dict) else None
        if not rgb_uri or not isinstance(rgb_uri, str):
            continue
        img_path = Path(rgb_uri)
        if not img_path.exists():
            # If URIs are relative to episode dir, try resolving.
            rel = (episode_path.parent / rgb_uri).resolve()
            img_path = rel if rel.exists() else img_path
        if not img_path.exists():
            continue

        masks, boxes_xyxy, scores = _segment_one_image(
            model=model,
            processor=processor,
            image_path=img_path,
            prompt=args.prompt,
            threshold=args.threshold,
            device=device,
        )

        seg_masks: List[Dict[str, Any]] = []
        seg_boxes: List[Dict[str, Any]] = []
        for i, mask in enumerate(masks):
            out_name = f"{frame_id or episode_path.stem}_sam3_mask_{idx:06d}_{i:03d}.png"
            out_path = blobs_dir / out_name
            _ensure_png_write(mask, out_path)
            seg_masks.append(
                {
                    "uri": str(out_path),
                    "frame_id": str(frame_id or ""),
                    "format": "png",
                    "instance_id": int(i),
                    "score": float(scores[i]) if i < len(scores) else 0.0,
                }
            )
        for i, box in enumerate(boxes_xyxy):
            if not isinstance(box, list) or len(box) != 4:
                continue
            seg_boxes.append(
                {
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "instance_id": int(i),
                    "score": float(scores[i]) if i < len(scores) else 0.0,
                }
            )

        obs["segmentation"] = {
            "model": str(args.model),
            "prompt": str(args.prompt),
            "masks": seg_masks,
            "boxes_xyxy": seg_boxes,
            "notes": "Generated offline via enrich_episode_sam3.py",
        }

    # Write back episode
    _save_json(episode_path, episode)
    print(f"✅ Wrote SAM3-enriched episode: {episode_path}")


if __name__ == "__main__":
    main()


