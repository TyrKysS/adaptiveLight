"""
Brightness→lux calibration sweep for RL agent warm-start.

Runs a one-time sweep across brightness levels, stores the resulting curve in
/data/rl_calibration.json, and provides bidirectional interpolation so any
target lux maps immediately to an approximate brightness without RL exploration.
"""

import json
import os

CALIB_PATH   = "/data/rl_calibration.json"
CALIB_STEPS  = [0, 20, 40, 60, 80, 100]   # brightness % measured in order
CALIB_SETTLE = 5.0                          # seconds to wait after each brightness change


class CalibrationData:
    """Brightness→lux curve with bidirectional linear interpolation."""

    def __init__(self, points: list[dict], timestamp: str = "") -> None:
        # points: [{"brightness": int, "lux": float}, ...]
        self.points    = sorted(points, key=lambda p: p["brightness"])
        self.timestamp = timestamp

    # ── interpolation ──────────────────────────────────────────────────────────

    def brightness_for_lux(self, target_lux: float) -> float:
        """Return interpolated brightness % that should produce target_lux."""
        by_lux = sorted(self.points, key=lambda p: p["lux"])
        if not by_lux:
            return 50.0
        if target_lux <= by_lux[0]["lux"]:
            return float(by_lux[0]["brightness"])
        if target_lux >= by_lux[-1]["lux"]:
            return float(by_lux[-1]["brightness"])
        for lo, hi in zip(by_lux, by_lux[1:]):
            if lo["lux"] <= target_lux <= hi["lux"]:
                span = hi["lux"] - lo["lux"]
                if span == 0:
                    return float(lo["brightness"])
                t = (target_lux - lo["lux"]) / span
                return lo["brightness"] + t * (hi["brightness"] - lo["brightness"])
        return 50.0

    def lux_for_brightness(self, brightness_pct: float) -> float:
        """Return interpolated lux expected at the given brightness %."""
        pts = self.points
        if not pts:
            return 0.0
        if brightness_pct <= pts[0]["brightness"]:
            return float(pts[0]["lux"])
        if brightness_pct >= pts[-1]["brightness"]:
            return float(pts[-1]["lux"])
        for lo, hi in zip(pts, pts[1:]):
            if lo["brightness"] <= brightness_pct <= hi["brightness"]:
                span = hi["brightness"] - lo["brightness"]
                if span == 0:
                    return float(lo["lux"])
                t = (brightness_pct - lo["brightness"]) / span
                return lo["lux"] + t * (hi["lux"] - lo["lux"])
        return 0.0

    @property
    def max_lux(self) -> float:
        return max((p["lux"] for p in self.points), default=0.0)

    # ── persistence ────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {"points": self.points, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationData":
        return cls(d["points"], d.get("timestamp", ""))


# ── module-level cache ─────────────────────────────────────────────────────────

_calibration: CalibrationData | None = None


def load_calibration() -> CalibrationData | None:
    global _calibration
    if _calibration is not None:
        return _calibration
    try:
        with open(CALIB_PATH) as fh:
            d = json.load(fh)
        _calibration = CalibrationData.from_dict(d)
        return _calibration
    except Exception:
        return None


def save_calibration(calib: CalibrationData) -> None:
    global _calibration
    try:
        os.makedirs("/data", exist_ok=True)
        with open(CALIB_PATH, "w") as fh:
            json.dump(calib.to_dict(), fh, indent=2)
        _calibration = calib
    except Exception:
        pass


def invalidate_cache() -> None:
    """Force next load_calibration() to re-read the file."""
    global _calibration
    _calibration = None
