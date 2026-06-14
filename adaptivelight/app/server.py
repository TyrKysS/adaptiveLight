import asyncio
import json
import os
import threading
import time
from datetime import datetime

import aiohttp
import numpy as np
import requests
from rl_agent import ACTIONS, RLAgent
from flask import Flask, jsonify, render_template
from flask import request as freq

app = Flask(__name__)

HA_API    = "http://supervisor/core/api"
HA_WS_URL = "ws://supervisor/core/websocket"
CONFIG_FILE = "/data/adaptivelight.json"

DEFAULT_CONFIG = {
    "selected_lights": [],
    "selected_lux_sensors": [],
    "selected_motion_sensors": [],
    "lux_threshold": 50,
    "automation_enabled": False,
    "auto_turn_off": False,
    # RL regulation
    "rl_enabled": False,
    "rl_target_lux": 100,
    "rl_input_sensors": [],
    "rl_output_lights": [],
    "rl_night_only": False,
    "rl_sun_threshold": 0.0,   # degrees; RL runs only when sun elevation < this
}

# ── Config ────────────────────────────────────────────────────────────────────

_cfg_cache: dict | None = None
_cfg_cache_ts: float = 0.0


def load_config() -> dict:
    global _cfg_cache, _cfg_cache_ts
    if _cfg_cache is None or (time.monotonic() - _cfg_cache_ts) > 5:
        try:
            with open(CONFIG_FILE) as f:
                _cfg_cache = {**DEFAULT_CONFIG, **json.load(f)}
        except Exception:
            _cfg_cache = dict(DEFAULT_CONFIG)
        _cfg_cache_ts = time.monotonic()
    return _cfg_cache


def save_config(cfg: dict) -> None:
    global _cfg_cache, _cfg_cache_ts
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    _cfg_cache = dict(cfg)
    _cfg_cache_ts = time.monotonic()


# ── HA REST helpers (for UI data + manual trigger) ────────────────────────────

def _rest_headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ.get('HA_TOKEN', '')}",
        "Content-Type": "application/json",
    }


def ha_get(path: str):
    try:
        r = requests.get(f"{HA_API}{path}", headers=_rest_headers(), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        app.logger.error("HA GET %s: %s", path, exc)
        return None


def ha_post(path: str, data: dict) -> bool:
    try:
        r = requests.post(f"{HA_API}{path}", headers=_rest_headers(), json=data, timeout=10)
        r.raise_for_status()
        return True
    except Exception as exc:
        app.logger.error("HA POST %s: %s", path, exc)
        return False


# ── Shared automation status ──────────────────────────────────────────────────

_status_lock = threading.Lock()
_auto_status: dict = {
    "last_run": None,
    "action": None,
    "lux": None,
    "threshold": None,
    "trigger": None,
    "error": None,
    "ws_connected": False,
}


def _set_status(**kw) -> None:
    with _status_lock:
        _auto_status.update(kw)


# ── RL agent singleton ────────────────────────────────────────────────────────

_rl_agent: "RLAgent | None" = None
_rl_brightness: float = 50.0   # brightness (0-100 %) tracked by RL
_rl_last_action_ts: float = 0.0
_rl_action_cooldown: float = 20.0  # seconds between RL actions
_rl_lock = threading.Lock()
_sun_elevation: float = 90.0   # cached from sun.sun; 90 = assume day until first update


def _get_rl_agent() -> RLAgent:
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = RLAgent()
    return _rl_agent


def _init_sun_elevation() -> None:
    global _sun_elevation
    time.sleep(5)  # wait for HA to be ready
    state = ha_get("/states/sun.sun")
    if state:
        try:
            _sun_elevation = float(state.get("attributes", {}).get("elevation", 90.0))
            app.logger.info("Sun elevation initialized: %.1f°", _sun_elevation)
        except (ValueError, TypeError):
            pass


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


# ── Manual REST-based trigger ("Run now" button) ──────────────────────────────

def run_automation_once() -> None:
    cfg = load_config()
    if not cfg["automation_enabled"]:
        return

    sensors = cfg["selected_lux_sensors"]
    lights  = cfg["selected_lights"]
    thr     = float(cfg["lux_threshold"])
    auto_off = cfg["auto_turn_off"]

    if not sensors or not lights:
        _set_status(last_run=_now(), error="Nejsou vybrány entity", action=None)
        return

    states = ha_get("/states")
    if states is None:
        _set_status(error="Nelze načíst stavy z HA")
        return

    sm = {s["entity_id"]: s for s in states}
    vals = []
    for sid in sensors:
        if sid in sm:
            try:
                vals.append(float(sm[sid]["state"]))
            except (ValueError, TypeError):
                pass

    if not vals:
        _set_status(last_run=_now(), error="Lux senzory nemají platné hodnoty", action=None)
        return

    avg = sum(vals) / len(vals)
    action = None
    if avg < thr:
        for lid in lights:
            ha_post("/services/light/turn_on", {"entity_id": lid})
        action = "turn_on"
    elif auto_off:
        for lid in lights:
            ha_post("/services/light/turn_off", {"entity_id": lid})
        action = "turn_off"

    _set_status(last_run=_now(), action=action, lux=round(avg, 1),
                threshold=thr, error=None, trigger="manuální spuštění")


# ── WebSocket automation (real-time, event-driven) ────────────────────────────

async def _apply_rule(ws, entity_id: str, lux_val: float, mid: list) -> None:
    cfg = load_config()
    if not cfg["automation_enabled"]:
        return

    lights   = cfg["selected_lights"]
    thr      = float(cfg["lux_threshold"])
    auto_off = cfg["auto_turn_off"]

    if not lights:
        return

    if lux_val < thr:
        for lid in lights:
            mid[0] += 1
            await ws.send_json({
                "id": mid[0], "type": "call_service",
                "domain": "light", "service": "turn_on",
                "service_data": {"entity_id": lid},
            })
        _set_status(last_run=_now(), action="turn_on", lux=round(lux_val, 1),
                    threshold=thr, error=None, trigger=entity_id)

    elif auto_off:
        for lid in lights:
            mid[0] += 1
            await ws.send_json({
                "id": mid[0], "type": "call_service",
                "domain": "light", "service": "turn_off",
                "service_data": {"entity_id": lid},
            })
        _set_status(last_run=_now(), action="turn_off", lux=round(lux_val, 1),
                    threshold=thr, error=None, trigger=entity_id)


async def _apply_rl(ws, entity_id: str, lux_val: float, mid: list) -> None:
    """RL brightness regulation — called from WebSocket event loop."""
    global _rl_brightness, _rl_last_action_ts
    cfg    = load_config()
    lights = cfg.get("rl_output_lights", [])
    target = float(cfg.get("rl_target_lux", 100))

    if not lights:
        return

    now = time.monotonic()
    if now - _rl_last_action_ts < _rl_action_cooldown:
        return

    if cfg.get("rl_night_only", False):
        if _sun_elevation > float(cfg.get("rl_sun_threshold", 0.0)):
            return

    agent = _get_rl_agent()
    with _rl_lock:
        cur_br = _rl_brightness

    state = RLAgent.build_state(lux_val, target, cur_br,
                                agent.prev_lux, agent.prev_action_idx)

    if agent.prev_state is not None:
        reward            = RLAgent.compute_reward(lux_val, target, agent.prev_lux)
        agent.last_reward = reward
        agent.remember(agent.prev_state, agent.prev_action_idx, reward, state)
        agent.replay()

    action_idx = agent.act(state)
    delta      = ACTIONS[action_idx]
    with _rl_lock:
        new_br        = float(np.clip(_rl_brightness + delta, 5, 100))
        _rl_brightness = new_br
    brightness_255 = int(new_br / 100 * 255)

    for lid in lights:
        mid[0] += 1
        await ws.send_json({
            "id": mid[0], "type": "call_service",
            "domain": "light", "service": "turn_on",
            "service_data": {"entity_id": lid, "brightness": brightness_255},
        })

    agent.prev_state      = state
    agent.prev_action_idx = action_idx
    agent.prev_lux        = lux_val
    agent.last_delta      = delta
    _rl_last_action_ts    = now

    _set_status(last_run=_now(), action=f"rl_br_{int(new_br)}",
                lux=round(lux_val, 1), threshold=target,
                trigger=entity_id, error=None)


async def _ws_session() -> None:
    token = os.environ.get("HA_TOKEN", "")
    mid = [0]

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(HA_WS_URL) as ws:

            # ── auth handshake ─────────────────────────────────────────────
            msg = await ws.receive_json()
            if msg.get("type") != "auth_required":
                raise RuntimeError(f"Unexpected WS message: {msg}")

            await ws.send_json({"type": "auth", "access_token": token})
            msg = await ws.receive_json()
            if msg.get("type") != "auth_ok":
                raise RuntimeError(f"WS auth failed: {msg}")

            # ── subscribe to state_changed ─────────────────────────────────
            mid[0] += 1
            await ws.send_json({
                "id": mid[0],
                "type": "subscribe_events",
                "event_type": "state_changed",
            })
            msg = await ws.receive_json()
            if not msg.get("success"):
                raise RuntimeError(f"Subscribe failed: {msg}")

            app.logger.info("WS: connected and listening for state_changed")
            _set_status(ws_connected=True, error=None)

            # ── event loop ─────────────────────────────────────────────────
            async for raw in ws:
                if raw.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(raw.data)

                    if data.get("type") != "event":
                        continue

                    event = data.get("event", {})
                    if event.get("event_type") != "state_changed":
                        continue

                    new_state = event.get("data", {}).get("new_state")
                    if not new_state:
                        continue

                    eid = new_state.get("entity_id", "")

                    # Track sun elevation (used by night-only RL mode)
                    if eid == "sun.sun":
                        global _sun_elevation
                        try:
                            _sun_elevation = float(
                                new_state.get("attributes", {}).get("elevation", _sun_elevation)
                            )
                        except (ValueError, TypeError):
                            pass
                        continue

                    cfg = load_config()

                    # RL path takes priority when enabled
                    if cfg.get("rl_enabled") and eid in cfg.get("rl_input_sensors", []):
                        try:
                            lux_val = float(new_state["state"])
                        except (ValueError, TypeError):
                            pass
                        else:
                            await _apply_rl(ws, eid, lux_val, mid)
                        continue

                    # Threshold path (existing)
                    if eid not in cfg.get("selected_lux_sensors", []):
                        continue

                    try:
                        lux_val = float(new_state["state"])
                    except (ValueError, TypeError):
                        continue

                    await _apply_rule(ws, eid, lux_val, mid)

                elif raw.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    raise RuntimeError(f"WS closed ({raw.type})")


async def _ws_loop() -> None:
    while True:
        try:
            await _ws_session()
        except Exception as exc:
            app.logger.warning("WS: %s — reconnect in 10 s", exc)
            _set_status(ws_connected=False, error=f"WS přerušeno: {exc}")
        await asyncio.sleep(10)


def _start_ws_thread() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_ws_loop())


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/entities")
def entities():
    states = ha_get("/states")
    if states is None:
        return jsonify({"error": "Nelze se připojit k Home Assistant API"}), 503

    lights, motion, lux = [], [], []

    for s in states:
        eid  = s["entity_id"]
        attr = s["attributes"]
        name = attr.get("friendly_name", eid)

        if eid.startswith("light."):
            br = attr.get("brightness")
            lights.append({
                "entity_id": eid, "state": s["state"], "friendly_name": name,
                "brightness_pct": round(br / 2.55) if br is not None else None,
                "rgb_color": attr.get("rgb_color"),
                "last_changed": s.get("last_changed", ""),
            })

        elif eid.startswith("binary_sensor.") and attr.get("device_class") in (
                "motion", "occupancy", "presence", "moving"):
            motion.append({
                "entity_id": eid, "state": s["state"], "friendly_name": name,
                "device_class": attr.get("device_class", ""),
                "last_changed": s.get("last_changed", ""),
            })

        elif eid.startswith("sensor.") and (
                attr.get("device_class") == "illuminance"
                or attr.get("unit_of_measurement") in ("lx", "lux")):
            lux.append({
                "entity_id": eid, "state": s["state"], "friendly_name": name,
                "unit": attr.get("unit_of_measurement") or "lx",
                "last_changed": s.get("last_changed", ""),
            })

    sun_ent = next((s for s in states if s["entity_id"] == "sun.sun"), None)
    sun = None
    if sun_ent:
        a = sun_ent["attributes"]
        sun = {
            "state": sun_ent["state"],
            "elevation": a.get("elevation"), "azimuth": a.get("azimuth"),
            "rising": a.get("rising"),
            "next_dawn": a.get("next_dawn", ""), "next_dusk": a.get("next_dusk", ""),
            "next_noon": a.get("next_noon", ""), "next_midnight": a.get("next_midnight", ""),
            "next_rising": a.get("next_rising", ""), "next_setting": a.get("next_setting", ""),
        }

    lights.sort(key=lambda x: (x["state"] != "on", x["friendly_name"].lower()))
    motion.sort(key=lambda x: (x["state"] != "on", x["friendly_name"].lower()))
    lux.sort(key=lambda x: x["friendly_name"].lower())

    return jsonify({
        "lights": lights, "motion_sensors": motion, "lux_sensors": lux, "sun": sun,
        "counts": {"lights": len(lights), "motion": len(motion), "lux": len(lux)},
        "timestamp": _now(),
    })


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify(load_config())


@app.route("/api/config", methods=["POST"])
def post_config():
    body = freq.get_json(silent=True)
    if not body:
        return jsonify({"error": "Chybí JSON tělo"}), 400
    cfg = load_config()
    for k, v in body.items():
        if k in DEFAULT_CONFIG:
            cfg[k] = v
    save_config(cfg)
    threading.Thread(target=run_automation_once, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/automation/status")
def automation_status():
    cfg = load_config()
    with _status_lock:
        return jsonify({**_auto_status, "enabled": cfg.get("automation_enabled", False)})


@app.route("/api/automation/run", methods=["POST"])
def automation_run():
    threading.Thread(target=run_automation_once, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/rl/status")
def rl_status():
    cfg      = load_config()
    agent    = _get_rl_agent()
    night    = cfg.get("rl_night_only", False)
    sun_thr  = float(cfg.get("rl_sun_threshold", 0.0))
    with _rl_lock:
        br = _rl_brightness
    return jsonify({
        **agent.info(),
        "enabled":            cfg.get("rl_enabled", False),
        "target_lux":         cfg.get("rl_target_lux", 100),
        "current_brightness": round(br, 1),
        "sun_elevation":      round(_sun_elevation, 1),
        "blocked_by_sun":     night and (_sun_elevation > sun_thr),
    })


@app.route("/api/rl/reset", methods=["POST"])
def rl_reset():
    _get_rl_agent().reset_model()
    try:
        os.remove("/data/rl_model.json")
    except FileNotFoundError:
        pass
    return jsonify({"ok": True})


# ── Boot ──────────────────────────────────────────────────────────────────────

threading.Thread(target=_start_ws_thread, daemon=True).start()
threading.Thread(target=_init_sun_elevation, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("INGRESS_PORT", 8099))
    app.run(host="0.0.0.0", port=port, threaded=True)
