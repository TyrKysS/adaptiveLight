import os
from datetime import datetime

import requests
from flask import Flask, jsonify, render_template

app = Flask(__name__)

HA_API = "http://supervisor/core/api"


def _headers():
    token = os.environ.get("HA_TOKEN", "")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def ha_get(path):
    try:
        resp = requests.get(f"{HA_API}{path}", headers=_headers(), timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        app.logger.error("HA API error %s: %s", path, exc)
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/entities")
def entities():
    states = ha_get("/states")
    if states is None:
        return jsonify({"error": "Nelze se připojit k Home Assistant API"}), 503

    lights = []
    for s in states:
        if not s["entity_id"].startswith("light."):
            continue
        brightness = s["attributes"].get("brightness")
        lights.append({
            "entity_id": s["entity_id"],
            "state": s["state"],
            "friendly_name": s["attributes"].get("friendly_name", s["entity_id"]),
            "brightness_pct": round(brightness / 2.55) if brightness is not None else None,
            "color_temp": s["attributes"].get("color_temp"),
            "rgb_color": s["attributes"].get("rgb_color"),
            "last_changed": s.get("last_changed", ""),
        })

    motion = []
    motion_classes = {"motion", "occupancy", "presence", "moving"}
    for s in states:
        if not s["entity_id"].startswith("binary_sensor."):
            continue
        dc = s["attributes"].get("device_class", "")
        if dc in motion_classes:
            motion.append({
                "entity_id": s["entity_id"],
                "state": s["state"],
                "friendly_name": s["attributes"].get("friendly_name", s["entity_id"]),
                "device_class": dc,
                "last_changed": s.get("last_changed", ""),
            })

    lux = []
    for s in states:
        if not s["entity_id"].startswith("sensor."):
            continue
        dc = s["attributes"].get("device_class", "")
        unit = s["attributes"].get("unit_of_measurement", "")
        if dc == "illuminance" or unit in ("lx", "lux"):
            lux.append({
                "entity_id": s["entity_id"],
                "state": s["state"],
                "friendly_name": s["attributes"].get("friendly_name", s["entity_id"]),
                "unit": unit or "lx",
                "last_changed": s.get("last_changed", ""),
            })

    sun_entity = next((s for s in states if s["entity_id"] == "sun.sun"), None)
    sun = None
    if sun_entity:
        attr = sun_entity["attributes"]
        sun = {
            "state": sun_entity["state"],
            "elevation": attr.get("elevation"),
            "azimuth": attr.get("azimuth"),
            "rising": attr.get("rising"),
            "next_dawn": attr.get("next_dawn", ""),
            "next_dusk": attr.get("next_dusk", ""),
            "next_noon": attr.get("next_noon", ""),
            "next_midnight": attr.get("next_midnight", ""),
            "next_rising": attr.get("next_rising", ""),
            "next_setting": attr.get("next_setting", ""),
        }

    lights.sort(key=lambda x: (x["state"] != "on", x["friendly_name"].lower()))
    motion.sort(key=lambda x: (x["state"] != "on", x["friendly_name"].lower()))
    lux.sort(key=lambda x: x["friendly_name"].lower())

    return jsonify({
        "lights": lights,
        "motion_sensors": motion,
        "lux_sensors": lux,
        "sun": sun,
        "counts": {
            "lights": len(lights),
            "motion": len(motion),
            "lux": len(lux),
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


if __name__ == "__main__":
    port = int(os.environ.get("INGRESS_PORT", 8099))
    app.run(host="0.0.0.0", port=port, threaded=True)
