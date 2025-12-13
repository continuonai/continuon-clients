"""
Hailo vision worker (subprocess-safe).

This module is designed to be executed as a separate process so that any
Hailo SDK instability (including potential segfaults) does not crash the main
Continuon Brain runtime process.

Protocol:
- Read JPEG bytes from stdin
- Run Hailo inference on the installed HEF
- Emit a single JSON object to stdout
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


def _json_out(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, default=str))
    sys.stdout.flush()


def _infer_hailo(jpeg_bytes: bytes, hef_path: Path, topk: int) -> Dict[str, Any]:
    import numpy as np  # type: ignore

    try:
        import hailo_platform as hailo  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"hailo_platform import failed: {exc}"}

    if not hef_path.exists():
        return {"ok": False, "error": f"hef not found: {hef_path}"}

    # Use the stable HEF+VStreams API (PCIe) instead of infer_model.configure(),
    # which can segfault on teardown in some environments.
    hef = hailo.HEF(str(hef_path))
    vdevice = hailo.VDevice()
    cfg_params = hailo.ConfigureParams.create_from_hef(hef, interface=hailo.HailoStreamInterface.PCIe)
    net_group = vdevice.configure(hef, cfg_params)[0]

    input_infos = list(hef.get_input_vstream_infos())
    output_infos = list(hef.get_output_vstream_infos())
    if not input_infos or not output_infos:
        return {"ok": False, "error": "hef missing vstream infos"}

    inp0 = input_infos[0]
    out0 = output_infos[0]
    inp_name = inp0.name
    inp_shape = tuple(int(x) for x in inp0.shape)
    out_name = out0.name
    out_shape = tuple(int(x) for x in out0.shape)
    inp_type = getattr(getattr(inp0, "format", None), "type", None)
    out_type = getattr(getattr(out0, "format", None), "type", None)

    # Many Hailo CV HEFs expect NHWC with shape (H, W, C).
    if len(inp_shape) == 3:
        h, w, c = inp_shape
        size_wh = (w, h)
    elif len(inp_shape) == 4:
        # Try to interpret as NHWC; if it isn't, downstream should override.
        _, h, w, c = inp_shape
        size_wh = (w, h)
    else:
        return {"ok": False, "error": f"unsupported input rank: {inp_shape}"}

    # Decode/resize JPEG to the HEF input size.
    from PIL import Image  # type: ignore

    im = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    im = im.resize(size_wh, Image.BILINEAR)
    img = np.asarray(im)

    # Prepare input buffer.
    # Use HEF-declared format type when available; most Hailo CV HEFs are UINT8.
    if str(inp_type).endswith("UINT8"):
        img_f = img.astype(np.uint8, copy=False)
        in_dtype = np.uint8
    else:
        img_f = img.astype(np.float32, copy=False) / 255.0
        in_dtype = np.float32

    # InferVStreams expects explicit batch dimension; do not let H be interpreted as batch.
    img_f = img_f.reshape((1,) + tuple(img_f.shape))

    input_buffer = np.array(img_f, dtype=in_dtype, copy=True, order="C")
    if str(out_type).endswith("UINT8"):
        out_format_type = hailo.FormatType.UINT8
    else:
        out_format_type = hailo.FormatType.FLOAT32
    if str(inp_type).endswith("UINT8"):
        in_format_type = hailo.FormatType.UINT8
    else:
        in_format_type = hailo.FormatType.FLOAT32

    in_params = hailo.InputVStreamParams.make_from_network_group(net_group, format_type=in_format_type)
    out_params = hailo.OutputVStreamParams.make_from_network_group(net_group, format_type=out_format_type)

    ng_params = net_group.create_params()
    with net_group.activate(ng_params):
        with hailo.InferVStreams(net_group, in_params, out_params) as infer:
            out = infer.infer({inp_name: input_buffer})

    # Output comes batched (1, ...); flatten for top-k.
    out_arr = out.get(out_name)
    if out_arr is None:
        # Fall back to first key if name mismatch.
        key0 = next(iter(out.keys()))
        out_arr = out[key0]
        out_name = key0
    logits = np.asarray(out_arr).astype(np.float32, copy=False).reshape(-1)
    if logits.size == 0:
        return {"ok": False, "error": "empty output"}

    k = max(1, min(int(topk), min(10, int(logits.size))))
    top_idx = np.argsort(-logits)[:k]
    top = [{"index": int(i), "score": float(logits[int(i)])} for i in top_idx]

    return {
        "ok": True,
        "hef_path": str(hef_path),
        "input": {"name": inp_name, "shape": (1,) + inp_shape},
        "output": {"name": out_name, "shape": (1,) + out_shape},
        "topk": top,
        "logits_summary": {"min": float(logits.min()), "max": float(logits.max())},
    }


def main() -> None:
    topk = int(os.environ.get("CONTINUON_HAILO_TOPK", "5") or 5)
    hef = Path(os.environ.get("CONTINUON_HAILO_HEF", "/opt/continuonos/brain/model/base_model/model.hef"))
    jpeg = sys.stdin.buffer.read()
    if not jpeg:
        _json_out({"ok": False, "error": "no stdin jpeg bytes"})
        return
    try:
        res = _infer_hailo(jpeg, hef_path=hef, topk=topk)
        _json_out(res)
    except Exception as exc:  # noqa: BLE001
        _json_out({"ok": False, "error": str(exc)})


if __name__ == "__main__":
    main()


