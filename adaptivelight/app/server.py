import json
import os
import threading
import time
from datetime import datetime

import requests
from flask import Flask, jsonify, render_template
from flask import request as freq

app = Flask(__name__)

HA_API = "http://supervisor/core/api"
CONFIG_FILE = "/data/adaptivelight.json"

DEFAULT_CONFIG = {
    "selected_lights": [],
    "selected_lux_sensors": [],
    "selected_motion_sensors": [],
    "lux_threshold": 50,
    "automation_enabled": False,
    "auto_turn_off": False,
}

# ── HA helpers ────────────────────────────────────────────────────────────────

def _headers():
    token = os.environ.get("HA_TOKEN", "")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def ha_get(path):
    try:
        r = requests.get(f"{HA_API}{path}", headers=_headers(), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        app.logger.error("HA GET %s: %s", path, exc)
        return None


def ha_post(path, data):
    try:
        r = requests.post(f"{HA_API}{path}", headers=_headers(), json=data, timeout=10)
        r.raise_for_status()
        return True
    except Exception as exc:
        app.logger.error("HA POST %s: %s", path, exc)
        return False


# ── Config persistence ────────────────────────────────────────────────────────

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def save_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


# ── Automation engine ─────────────────────────────────────────────────────────

_status_lock = threading.Lock()
_auto_status = {
    "last_run": None,
    "action": None,
    "avg_lux": None,
    "threshold": None,
    "error": None,
}


def run_automation_once():
    cfg = load_config()
    if not cfg.get("automation_enabled"):
        return

    lux_sensors = cfg.get("selected_lux_sensors", [])
    lights = cfg.get("selected_lights", [])
    threshold = float(cfg.get("lux_threshold", 50))
    auto_turn_off = cfg.get("auto_turn_off", False)

    if not lux_sensors or not lights:
        with _status_lock:
            _auto_status.update({"last_run": _now(), "action": None,
                                  "avg_lux": None, "threshold": threshold,
                                  "error": "Nejsou vybrány entity"})
        return

    states = ha_get("/states")
    if states is None:
        with _status_lock:
            _auto_status["error"] = "Nelze načíst stavy z HA"
        return

    state_map = {s["entity_id"]: s for s in states}

    lux_values = []
    for sid in lux_sensors:
        if sid in state_map:
            try:
                lux_values.append(float(state_map[sid]["state"]))
            except (ValueError, TypeError):
                pass

    if not lux_values:
        with _status_lock:
            _auto_status.update({"last_run": _now(), "action": None,
                                  "avg_lux": None, "threshold": threshold,
                                  "error": "Lux senzory nemají platné hodnoty"})
        return

    avg_lux = sum(lux_values) / len(lux_values)
    action = None

    if avg_lux < threshold:
        for lid in lights:
            ha_post("/services/light/turn_on", {"entity_id": lid})
        action = "turn_on"
    elif auto_turn_off:
        for lid in lights:
            ha_post("/services/light/turn_off", {"entity_id": lid})
        action = "turn_off"

    with _status_lock:
        _auto_status.update({
            "last_run": _now(),
            "action": action,
            "avg_lux": round(avg_lux, 1),
            "threshold": threshold,
            "error": None,
        })


def _now():
    return datetime.utcnow().isoformat() + "Z"


def _automation_loop():
    while True:
        try:
            run_automation_once()
        except Exception as exc:
            app.logger.error("Automation loop error: %s", exc)
        time.sleep(30)


# ── Routes ────────────────────────────────────────────────────────────────────

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
        eid = s["entity_id"]
        attr = s["attributes"]
        name = attr.get("friendly_name", eid)

        if eid.startswith("light."):
            brightness = attr.get("brightness")
            lights.append({
                "entity_id": eid, "state": s["state"], "friendly_name": name,
                "brightness_pct": round(brightness / 2.55) if brightness is not None else None,
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

    sun_entity = next((s for s in states if s["entity_id"] == "sun.sun"), None)
    sun = None
    if sun_entity:
        a = sun_entity["attributes"]
        sun = {
            "state": sun_entity["state"],
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
    allowed = set(DEFAULT_CONFIG.keys())
    for k, v in body.items():
        if k in allowed:
            cfg[k] = v
    save_config(cfg)
    # Immediately apply automation after save
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


# ── Start ─────────────────────────────────────────────────────────────────────

threading.Thread(target=_automation_loop, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("INGRESS_PORT", 8099))
    app.run(host="0.0.0.0", port=port, threaded=True)
