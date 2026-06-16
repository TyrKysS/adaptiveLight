import asyncio
import json
import os
import threading
import time
from datetime import datetime

import aiohttp
import numpy as np
import requests
from rl_agent import ACTIONS, MODEL_PATH, RLAgent
from rl_calibration import (CalibrationData, load_calibration, save_calibration,
                             invalidate_cache, CALIB_STEPS, CALIB_SETTLE)
from flask import Flask, jsonify, render_template, Response
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
    "rl_action_cooldown":  3.0, # seconds between successive RL brightness adjustments
    "rl_lux_tolerance":    0.5, # dead band ± lux around target; no command issued inside
    # Simulation training
    "rl_sim_episodes":     1000, # number of simulation episodes
    "rl_sim_steps_per_ep":   30, # environment steps per episode
    "rl_sim_noise_std":     2.0, # Gaussian lux noise std dev for digital twin
    "rl_sim_goals":          [], # explicit goal lux values; empty = auto from calibration
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
_rl_lock = threading.Lock()
_sun_elevation: float = 90.0   # cached from sun.sun; 90 = assume day until first update
_rl_boundary_steps: int = 0    # consecutive steps stuck at min/max brightness
_rl_light_is_off:   bool = False  # True while lamp is off due to ambient lux override

_calib_lock = threading.Lock()
_calib_status: dict = {
    "running":            False,
    "step":               0,
    "total":              len(CALIB_STEPS),
    "current_brightness": None,
    "points":             [],
    "error":              None,
    "timestamp":          None,
}

_sim_lock = threading.Lock()
_sim_status: dict = {
    "running":    False,
    "episode":    0,
    "total":      0,
    "epsilon":    0.0,
    "done":       False,
    "error":      None,
    "trained_at": None,
}


def _get_rl_agent() -> RLAgent:
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = RLAgent()
    return _rl_agent


def _set_rl_brightness_rest(brightness_pct: float, lights: list) -> None:
    """Command lights via REST and update the RL brightness tracker."""
    global _rl_brightness
    new_br = float(np.clip(brightness_pct, 5, 100))
    with _rl_lock:
        _rl_brightness = new_br
    b255 = int(new_br / 100 * 255)
    for lid in lights:
        ha_post("/services/light/turn_on", {"entity_id": lid, "brightness": b255})


def _run_calibration_thread() -> None:
    """Sweep brightness 0→100 %, measure lux at each step, save curve."""
    global _calib_status
    cfg     = load_config()
    lights  = cfg.get("rl_output_lights", [])
    sensors = cfg.get("rl_input_sensors", [])

    if not lights or not sensors:
        with _calib_lock:
            _calib_status.update(running=False,
                                  error="Nejsou nastavena výstupní světla nebo vstupní senzory RL")
        return

    with _calib_lock:
        _calib_status.update(running=True, step=0, points=[], error=None,
                              current_brightness=None, timestamp=None)

    points = []
    try:
        for i, pct in enumerate(CALIB_STEPS):
            with _calib_lock:
                _calib_status.update(step=i + 1, current_brightness=pct)

            if pct == 0:
                for lid in lights:
                    ha_post("/services/light/turn_off", {"entity_id": lid})
            else:
                b255 = int(pct / 100 * 255)
                for lid in lights:
                    ha_post("/services/light/turn_on",
                            {"entity_id": lid, "brightness": b255})

            time.sleep(CALIB_SETTLE)

            lux_vals = []
            for sid in sensors:
                st = ha_get(f"/states/{sid}")
                if st:
                    try:
                        lux_vals.append(float(st["state"]))
                    except (ValueError, TypeError):
                        pass

            if lux_vals:
                avg = sum(lux_vals) / len(lux_vals)
                pt  = {"brightness": pct, "lux": round(avg, 2)}
                points.append(pt)
                with _calib_lock:
                    _calib_status["points"] = list(points)
                app.logger.info("Calib %d%%: %.2f lx", pct, avg)

        ts    = _now()
        calib = CalibrationData(points, timestamp=ts)
        save_calibration(calib)

        with _calib_lock:
            _calib_status.update(running=False, step=len(CALIB_STEPS),
                                  error=None, timestamp=ts)

        # Immediately set lamp to curve brightness for current target
        cfg2   = load_config()
        target = float(cfg2.get("rl_target_lux", 100))
        _set_rl_brightness_rest(calib.brightness_for_lux(target), lights)

        # Auto-start simulation training on the freshly measured curve
        threading.Thread(target=_run_simulation_thread, daemon=True).start()

    except Exception as exc:
        with _calib_lock:
            _calib_status.update(running=False, error=str(exc))
        app.logger.error("Calibration failed: %s", exc)


def _run_simulation_thread() -> None:
    """Train the RL agent in simulation using the calibration curve as digital twin."""
    global _sim_status
    cfg   = load_config()
    calib = load_calibration()

    if not calib:
        with _sim_lock:
            _sim_status.update(running=False,
                               error="Kalibrace chybí — nejprve proveďte kalibraci")
        return

    with _calib_lock:
        if _calib_status["running"]:
            with _sim_lock:
                _sim_status.update(running=False,
                                   error="Kalibrace stále probíhá — počkejte na dokončení")
            return

    goals     = [float(g) for g in cfg.get("rl_sim_goals", []) if float(g) > 0]
    n_ep      = max(1, int(cfg.get("rl_sim_episodes",    1000)))
    steps_ep  = max(5, int(cfg.get("rl_sim_steps_per_ep",  30)))
    noise     = max(0.0, float(cfg.get("rl_sim_noise_std",  2.0)))

    with _sim_lock:
        _sim_status.update(running=True, episode=0, total=n_ep,
                           epsilon=0.6, done=False, error=None, trained_at=None)

    agent = _get_rl_agent()
    # Reset training state so simulation starts fresh while keeping live-mode flag until done
    agent.epsilon        = 0.60
    agent.epsilon_min    = 0.05
    agent.epsilon_decay  = 0.99
    agent.lr             = 5e-4
    agent.trained_by_sim = False
    agent.memory.clear()
    agent.prev_state     = None
    agent.prev_lux       = None

    def _progress(ep: int, total: int, eps: float) -> None:
        with _sim_lock:
            _sim_status.update(episode=ep, total=total, epsilon=round(eps, 4))

    try:
        agent.simulate(calib, goals, n_ep, steps_ep, noise, _progress)
        ts = _now()
        with _sim_lock:
            _sim_status.update(running=False, episode=n_ep, done=True,
                               epsilon=0.0, trained_at=ts)
        app.logger.info("RL simulation done: %d episodes, %d total steps", n_ep, agent.steps)
    except Exception as exc:
        with _sim_lock:
            _sim_status.update(running=False, error=str(exc))
        app.logger.error("RL simulation failed: %s", exc)


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
    global _rl_brightness, _rl_last_action_ts, _rl_boundary_steps, _rl_light_is_off
    cfg    = load_config()
    lights = cfg.get("rl_output_lights", [])
    target = float(cfg.get("rl_target_lux", 100))

    if not lights:
        return

    now = time.monotonic()
    cooldown = float(cfg.get("rl_action_cooldown", 3.0))
    if now - _rl_last_action_ts < cooldown:
        return

    if cfg.get("rl_night_only", False):
        if _sun_elevation > float(cfg.get("rl_sun_threshold", 0.0)):
            return

    # Block RL while calibration sweep is running
    with _calib_lock:
        if _calib_status["running"]:
            return

    agent     = _get_rl_agent()
    tolerance = float(cfg.get("rl_lux_tolerance", 0.5))

    # ── Ambient override: stay off until lux drops back below target ──────────
    if _rl_light_is_off:
        if lux_val > target + tolerance:
            return  # ambient still above target — keep light off, skip cooldown bump
        # Lux fell below target: clear flag and fall through to warm-start
        _rl_light_is_off = False
        agent.prev_state = None
        agent.prev_lux   = None

    # ── Spike filter ──────────────────────────────────────────────────────────
    # Skip readings where lux changed more than 70 % of target in a single step
    # (sensor covered/uncovered, sudden manual light switch).  For target=50 this
    # is a 35-lux jump — larger than any lamp adjustment but smaller than full
    # coverage.  We slide prev_lux forward so the *next* event starts cleanly.
    if agent.prev_lux is not None:
        spike_thr = max(target * 0.7, 20.0)
        if abs(lux_val - agent.prev_lux) > spike_thr:
            app.logger.warning("RL: lux spike %.1f→%.1f (thr %.0f), skipping step",
                               agent.prev_lux, lux_val, spike_thr)
            agent.prev_lux = lux_val
            return

    in_band = abs(lux_val - target) <= tolerance

    # ── Stuck-at-boundary detection ───────────────────────────────────────────
    # If the agent has been at minimum brightness while lux is still well below
    # target for several consecutive steps, something is wrong (e.g. daytime ambient
    # less than target, or policy stuck). Force a warm-start to re-anchor.
    with _rl_lock:
        cur_br = _rl_brightness

    if cur_br <= 5.5 and lux_val < target * 0.90 and not in_band:
        _rl_boundary_steps += 1
        if _rl_boundary_steps >= 6:
            app.logger.info("RL: stuck at min brightness (%.1f lux vs %.1f target), forcing warm-start",
                            lux_val, target)
            agent.prev_state = None
            agent.prev_lux   = None
            _rl_boundary_steps = 0
    else:
        _rl_boundary_steps = 0

    # ── Ambient overflow: turn off when lamp cannot reduce lux further ─────────
    # At minimum brightness the lamp cannot decrease lux. If ambient light alone
    # keeps lux above the setpoint, turn the light off and set the flag so the
    # warm-start below does not immediately switch it back on.
    if cur_br <= 5.5 and lux_val > target + tolerance:
        if not _rl_light_is_off:
            for lid in lights:
                mid[0] += 1
                await ws.send_json({
                    "id": mid[0], "type": "call_service",
                    "domain": "light", "service": "turn_off",
                    "service_data": {"entity_id": lid},
                })
            app.logger.info("RL: ambient lux %.1f exceeds target %.1f at min brightness → turn off",
                            lux_val, target)
            _rl_light_is_off   = True
            agent.prev_state   = None
            agent.prev_lux     = None
            _rl_boundary_steps = 0
            _rl_last_action_ts = now
            _set_status(last_run=_now(), action="rl_ambient_off",
                        lux=round(lux_val, 1), threshold=target,
                        trigger=entity_id, error=None)
        return

    # ── Warm-start ────────────────────────────────────────────────────────────
    # On first step (or after a forced reset), command the light to the
    # calibration-curve brightness so the *next* lux reading reflects an accurate
    # starting point. Return immediately — the agent acts on the following event.
    calib = load_calibration()
    if calib and agent.prev_state is None:
        approx_br = float(np.clip(calib.brightness_for_lux(target), 5, 100))
        with _rl_lock:
            _rl_brightness = approx_br
        b255 = int(approx_br / 100 * 255)
        for lid in lights:
            mid[0] += 1
            await ws.send_json({
                "id": mid[0], "type": "call_service",
                "domain": "light", "service": "turn_on",
                "service_data": {"entity_id": lid, "brightness": b255},
            })
        # Build a placeholder state so warm-start doesn't fire again next event
        placeholder = RLAgent.build_state(lux_val, target, approx_br, None, ACTIONS.index(0))
        agent.prev_state      = placeholder
        agent.prev_action_idx = ACTIONS.index(0)
        agent.prev_lux        = lux_val
        _rl_last_action_ts    = now
        app.logger.info("RL warm-start: target %.1f lx → %.0f%% brightness", target, approx_br)
        _set_status(last_run=_now(), action=f"rl_warmstart_{int(approx_br)}",
                    lux=round(lux_val, 1), threshold=target, trigger=entity_id, error=None)
        return

    with _rl_lock:
        cur_br = _rl_brightness

    state = RLAgent.build_state(lux_val, target, cur_br,
                                agent.prev_lux, agent.prev_action_idx)

    if agent.prev_state is not None:
        reward            = RLAgent.compute_reward(lux_val, target, agent.prev_lux, tolerance)
        agent.last_reward = reward
        agent.remember(agent.prev_state, agent.prev_action_idx, reward, state)
        agent.replay()

    action_idx = agent.act_guarded(state, lux_val, target)
    delta      = ACTIONS[action_idx]

    # Always update state tracking and cooldown
    agent.prev_state      = state
    agent.prev_action_idx = action_idx
    agent.prev_lux        = lux_val
    _rl_last_action_ts    = now

    # Inside dead band: skip brightness command, RL only learns
    if in_band:
        agent.last_delta = 0
        _set_status(last_run=_now(), action="rl_in_band",
                    lux=round(lux_val, 1), threshold=target,
                    trigger=entity_id, error=None)
        return

    with _rl_lock:
        new_br         = float(np.clip(_rl_brightness + delta, 5, 100))
        _rl_brightness = new_br
    brightness_255 = int(new_br / 100 * 255)

    for lid in lights:
        mid[0] += 1
        await ws.send_json({
            "id": mid[0], "type": "call_service",
            "domain": "light", "service": "turn_on",
            "service_data": {"entity_id": lid, "brightness": brightness_255},
        })

    agent.last_delta = delta
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
    old_target = cfg.get("rl_target_lux")
    for k, v in body.items():
        if k in DEFAULT_CONFIG:
            cfg[k] = v
    save_config(cfg)
    target_changed = cfg.get("rl_target_lux") != old_target
    if target_changed:
        agent = _get_rl_agent()
        agent.memory.clear()
        agent.prev_state = None
        agent.prev_lux   = None
        app.logger.info("RL target changed %s→%s: replay buffer flushed",
                        old_target, cfg["rl_target_lux"])
        calib = load_calibration()
        if calib:
            lights = cfg.get("rl_output_lights", [])
            _set_rl_brightness_rest(
                calib.brightness_for_lux(float(cfg["rl_target_lux"])), lights)
    threading.Thread(target=run_automation_once, daemon=True).start()
    return jsonify({"ok": True, "buffer_flushed": target_changed})


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
    cfg     = load_config()
    agent   = _get_rl_agent()
    night   = cfg.get("rl_night_only", False)
    sun_thr = float(cfg.get("rl_sun_threshold", 0.0))
    with _rl_lock:
        br = _rl_brightness
    with _calib_lock:
        calib_running = _calib_status["running"]
    with _sim_lock:
        sim_running = _sim_status["running"]
        sim_done    = _sim_status["done"]
    calib = load_calibration()
    return jsonify({
        **agent.info(),
        "enabled":            cfg.get("rl_enabled", False),
        "target_lux":         cfg.get("rl_target_lux", 100),
        "current_brightness": round(br, 1),
        "sun_elevation":      round(_sun_elevation, 1),
        "blocked_by_sun":     night and (_sun_elevation > sun_thr),
        "calibrated":         calib is not None,
        "calib_running":      calib_running,
        "calib_max_lux":      round(calib.max_lux, 1) if calib else None,
        "sim_running":        sim_running,
        "sim_done":           sim_done,
    })


@app.route("/api/rl/reset", methods=["POST"])
def rl_reset():
    _get_rl_agent().reset_model()
    try:
        os.remove("/data/rl_model.json")
    except FileNotFoundError:
        pass
    with _sim_lock:
        _sim_status.update(running=False, episode=0, total=0, epsilon=0.0,
                           done=False, error=None, trained_at=None)
    return jsonify({"ok": True})


@app.route("/api/rl/simulate", methods=["POST"])
def rl_simulate():
    with _sim_lock:
        if _sim_status["running"]:
            return jsonify({"error": "Simulace již probíhá"}), 409
    threading.Thread(target=_run_simulation_thread, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/rl/simulate/status")
def rl_simulate_status():
    with _sim_lock:
        return jsonify(dict(_sim_status))


@app.route("/api/rl/calibration")
def rl_calibration_get():
    with _calib_lock:
        status = dict(_calib_status)
    calib = load_calibration()
    return jsonify({
        "status":      status,
        "calibration": calib.to_dict() if calib else None,
    })


@app.route("/api/rl/calibrate", methods=["POST"])
def rl_calibrate():
    with _calib_lock:
        if _calib_status["running"]:
            return jsonify({"error": "Kalibrace již probíhá"}), 409
    threading.Thread(target=_run_calibration_thread, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/rl/target", methods=["POST"])
def rl_set_target():
    """Dedicated endpoint for external agents to set target lux instantly."""
    body = freq.get_json(silent=True)
    if not body or "target_lux" not in body:
        return jsonify({"error": "Chybí target_lux"}), 400
    new_target = float(body["target_lux"])
    cfg = load_config()
    old_target = cfg.get("rl_target_lux")
    cfg["rl_target_lux"] = new_target
    save_config(cfg)
    if new_target != old_target:
        agent = _get_rl_agent()
        agent.memory.clear()
        agent.prev_state = None
        agent.prev_lux   = None
        calib = load_calibration()
        if calib:
            lights = cfg.get("rl_output_lights", [])
            _set_rl_brightness_rest(calib.brightness_for_lux(new_target), lights)
    return jsonify({"ok": True, "target_lux": new_target})


@app.route("/api/rl/export")
def rl_export():
    """Bundle model weights, calibration curve and RL config for offline experiments."""
    bundle: dict = {
        "export_timestamp": _now(),
        "model":            None,
        "calibration":      None,
        "config":           None,
        "sim_status":       None,
    }

    try:
        with open(MODEL_PATH) as fh:
            bundle["model"] = json.load(fh)
    except FileNotFoundError:
        pass

    calib = load_calibration()
    if calib:
        bundle["calibration"] = calib.to_dict()

    cfg = load_config()
    bundle["config"] = {k: v for k, v in cfg.items() if k.startswith("rl_")}

    with _sim_lock:
        bundle["sim_status"] = dict(_sim_status)

    filename = f"adaptivelight_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    return Response(
        json.dumps(bundle, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Boot ──────────────────────────────────────────────────────────────────────

threading.Thread(target=_start_ws_thread, daemon=True).start()
threading.Thread(target=_init_sun_elevation, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("INGRESS_PORT", 8099))
    app.run(host="0.0.0.0", port=port, threaded=True)
