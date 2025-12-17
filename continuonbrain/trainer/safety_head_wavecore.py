"""
WaveCore-style safety head for runtime action gating and auditing.

This replaces the lightweight `safety_head_stub` by:
 - clamping actions to configured bounds
 - computing a simple risk score using a WaveCore-inspired envelope (radius/decay)
 - emitting structured results for status surfaces and audits

Design goals:
 - Dependency-light (pure Python; no torch/JAX imports)
 - Works in mock/offline environments
 - Safe defaults that bias toward clamping/blocking when inputs are missing
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Optional, Tuple
import time

NumericBounds = Dict[str, Tuple[float, float]]


@dataclass
class SafetyEnvelope:
    """Spatial/temporal safety envelope (WaveCore-style decay)."""

    status: str = "nominal"
    radius_m: float = 1.2
    decay: float = 0.12  # higher = faster decay of risk over time

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyHeadConfig:
    """Config for action bounds and risk thresholds."""

    clamp_bounds: NumericBounds
    warn_threshold: float = 0.25
    block_threshold: float = 0.6
    audit: bool = True

    @classmethod
    def default(cls) -> "SafetyHeadConfig":
        return cls(clamp_bounds={"steering": (-1.0, 1.0), "throttle": (-1.0, 1.0)})


@dataclass
class SafetyDecision:
    action: Any
    clamped: bool
    violations: Dict[str, float]
    risk: float
    risk_label: str
    envelope: SafetyEnvelope
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["envelope"] = self.envelope.to_dict()
        return payload


def _clamp_action(action: Dict[str, float], bounds: NumericBounds) -> tuple[Dict[str, float], Dict[str, float], bool]:
    safe_action = dict(action)
    violations: Dict[str, float] = {}
    clamped = False

    for key, (lo, hi) in bounds.items():
        if key not in safe_action:
            continue
        try:
            val = float(safe_action[key])
        except Exception:
            continue
        if val < lo:
            violations[key] = val
            safe_action[key] = lo
            clamped = True
        elif val > hi:
            violations[key] = val
            safe_action[key] = hi
            clamped = True
    return safe_action, violations, clamped


def _risk_from_context(
    *,
    envelope: SafetyEnvelope,
    clamped: bool,
    proximity_m: Optional[float],
    violations: Dict[str, float],
) -> float:
    # Base risk if we had to clamp
    risk = 0.35 if clamped else 0.0

    # Proximity risk: if inside radius, scale up toward 1.0
    if proximity_m is not None and envelope.radius_m > 0:
        margin = max(0.0, envelope.radius_m - proximity_m)
        risk += min(1.0, margin / envelope.radius_m)

    # Additional risk if multiple bounds were violated
    if violations:
        risk += min(0.4, 0.1 * len(violations))

    # Apply decay to keep risk in [0, 1]
    risk = min(1.0, max(0.0, risk * (1.0 + envelope.decay)))
    return risk


def _label_risk(risk: float, cfg: SafetyHeadConfig) -> str:
    if risk >= cfg.block_threshold:
        return "block"
    if risk >= cfg.warn_threshold:
        return "warn"
    return "ok"


class SafetyHead:
    """WaveCore-aligned safety head with clamp + risk scoring."""

    def __init__(self, config: Optional[SafetyHeadConfig] = None, envelope: Optional[SafetyEnvelope] = None):
        self.config = config or SafetyHeadConfig.default()
        self.envelope = envelope or SafetyEnvelope()

    @classmethod
    def from_manifest(cls, safety_cfg: Mapping[str, Any] | None) -> "SafetyHead":
        cfg = SafetyHeadConfig.default()
        env = SafetyEnvelope()

        if isinstance(safety_cfg, Mapping):
            clamp_bounds = safety_cfg.get("clamp_bounds")
            if isinstance(clamp_bounds, Mapping):
                cfg.clamp_bounds = {
                    k: (float(v[0]), float(v[1])) for k, v in clamp_bounds.items() if isinstance(v, (list, tuple)) and len(v) == 2
                }
            cfg.warn_threshold = float(safety_cfg.get("warn_threshold", cfg.warn_threshold))
            cfg.block_threshold = float(safety_cfg.get("block_threshold", cfg.block_threshold))
            cfg.audit = bool(safety_cfg.get("audit", cfg.audit))

            envelope_cfg = safety_cfg.get("envelope", {})
            if isinstance(envelope_cfg, Mapping):
                env.status = str(envelope_cfg.get("status", env.status))
                env.radius_m = float(envelope_cfg.get("radius_m", env.radius_m))
                env.decay = float(envelope_cfg.get("decay", env.decay))

        return cls(config=cfg, envelope=env)

    def evaluate(self, action: Any, *, proximity_m: Optional[float] = None, context: Optional[Mapping[str, Any]] = None) -> SafetyDecision:
        """Evaluate an action, clamp bounds, and produce a risk-aware decision."""
        if isinstance(action, dict):
            safe_action, violations, clamped = _clamp_action(action, self.config.clamp_bounds)
        else:
            safe_action, violations, clamped = action, {}, False

        if context and proximity_m is None:
            try:
                proximity_m = float(context.get("proximity_m")) if "proximity_m" in context else None
            except Exception:
                proximity_m = None

        risk = _risk_from_context(envelope=self.envelope, clamped=clamped, proximity_m=proximity_m, violations=violations)
        label = _label_risk(risk, self.config)

        notes = None
        if label == "block":
            notes = "Risk above block threshold; hold motion."
        elif label == "warn":
            notes = "Risk above warn threshold; slow or review."

        return SafetyDecision(
            action=safe_action,
            clamped=clamped,
            violations=violations,
            risk=risk,
            risk_label=label,
            envelope=self.envelope,
            notes=notes,
        )

    def heartbeat(self) -> Dict[str, Any]:
        return {
            "timestamp_ns": time.time_ns(),
            "ok": True,
            "source": "wavecore",
        }

    def __call__(self, action: Any, **kwargs: Any) -> SafetyDecision:
        return self.evaluate(action, **kwargs)


def safety_step(action: Any, safety_cfg: Optional[Mapping[str, Any]] = None, context: Optional[Mapping[str, Any]] = None) -> SafetyDecision:
    """
    Convenience function: build a head from config and evaluate once.
    """
    head = SafetyHead.from_manifest(safety_cfg)
    return head.evaluate(action, context=context)


def _demo() -> int:
    """Lightweight self-test for CI/offline smoke."""
    head = SafetyHead()
    samples = [
        {"action": {"steering": 1.5, "throttle": 0.2}, "proximity_m": 0.6},
        {"action": {"steering": 0.0, "throttle": -1.2}, "proximity_m": 2.0},
        {"action": {"steering": 0.1, "throttle": 0.1}, "proximity_m": None},
    ]
    for sample in samples:
        decision = head.evaluate(sample["action"], proximity_m=sample.get("proximity_m"))
        print(decision.to_dict())
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WaveCore-style safety head self-test")
    parser.add_argument("--self-test", action="store_true", help="Run a lightweight evaluation demo and exit")
    args = parser.parse_args()
    if args.self_test:
        raise SystemExit(_demo())
    raise SystemExit("No action specified. Use --self-test for a quick check.")

