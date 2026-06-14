# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Home Assistant add-on that displays a dashboard of lighting-relevant entities (lights, motion sensors, lux sensors, sun data) and provides basic lux-based automation. It is installed via the HA add-on store by pointing to this repo URL.

## Architecture

The add-on is a single Python/Flask app with no database and no build step.

- **`adaptivelight/app/server.py`** — Flask backend. Calls the HA Supervisor REST API (`http://supervisor/core/api`) for entity states and uses a persistent WebSocket connection (`ws://supervisor/core/websocket`) driven by `aiohttp` inside a daemon thread via `asyncio.run()` for real-time lux automation. Config is stored in `/data/adaptivelight.json` (inside the container).
- **`adaptivelight/app/templates/index.html`** — Entire frontend: HTML + CSS + vanilla JS in one file. Polls `/api/entities` every 30 s, RL stats every 10 s. Tabs: Přehled, Nastavení, RL Regulace.
- **`adaptivelight/app/rl_agent.py`** — Self-contained RL module (no ML framework). See section below.
- **`adaptivelight/app/rl_calibration.py`** — Brightness→lux calibration sweep. `CalibrationData` holds the measured curve and provides bidirectional linear interpolation (`brightness_for_lux`, `lux_for_brightness`). Curve is cached in module-level `_calibration` and persisted to `/data/rl_calibration.json`.
- **`adaptivelight/run.sh`** — Container entrypoint (bashio). Sets `HA_TOKEN` from `$SUPERVISOR_TOKEN`, then starts `server.py`.
- **`adaptivelight/config.yaml`** — Add-on manifest. Declares ingress on port 8099, requires both `hassio_api` and `homeassistant_api`.
- **`adaptivelight/build.yaml`** — Multi-arch base images (Python 3.11 / Alpine 3.18).
- **`repository.yaml`** — Marks this repo as an HA add-on repository.

## RL architecture (`rl_agent.py`)

Closed-loop brightness regulation via Q-learning. Triggered on every lux `state_changed` WebSocket event.

**Neural network:** 4 → 32 → 16 → 11 (ReLU hidden, linear output, He init, manual SGD backprop — pure numpy, no framework).

**State vector:** `[lux_error_norm, brightness_norm, lux_trend_norm, prev_action_norm]`
- `lux_error_norm` = `(current_lux − target_lux) / target_lux`, clipped to [−3, 3]
- `brightness_norm` = current RL-tracked brightness / 100
- `lux_trend_norm` = change since last reading / target_lux, clipped to [−1, 1]
- `prev_action_norm` = last brightness delta / 25

**Actions (11 discrete):** brightness delta in % — `[−25, −15, −10, −5, −2, 0, +2, +5, +10, +15, +25]`

**Reward:** `−|error_pct| × 2`, bonus `+2.0` at <5 % error, `+0.5` at <15 %, `+0.4` if improving vs. previous step.

**Training:** experience replay (buffer 4 000, batch 8), target network synced every 50 steps, ε-greedy (0.60 → 0.05 decay × 0.99/step), model auto-saved to `/data/rl_model.json` every 50 steps.

**Gate conditions:** RL only fires when `rl_enabled` is true and calibration is not running. If `rl_night_only` is true, it additionally requires sun elevation < `rl_sun_threshold` (degrees). `_sun_elevation` is a module-level float (default 90°, updated from WS `sun.sun` events). There is also a configurable cooldown (`rl_action_cooldown`, default 3 s) between successive RL brightness adjustments — stored in config and read at runtime by `_apply_rl`.

**Calibration warm-start:** On the first RL action per session (when `prev_state is None`), `_apply_rl` calls `load_calibration()` and uses `brightness_for_lux(target)` to jump `_rl_brightness` to the curve's suggested value instead of starting from 50 %. Changing `rl_target_lux` flushes the replay buffer, resets `prev_state`/`prev_lux`, and immediately commands the lamp to the calibration-curve brightness via `_set_rl_brightness_rest()`.

**Threading:** `RLAgent` is a singleton (`_get_rl_agent()`). Its internal `_lock` guards network weights. Brightness is tracked in module-level `_rl_brightness` guarded by `_rl_lock`. The RL path in the WS event loop takes priority over the threshold path — a sensor in `rl_input_sensors` is never passed to `_apply_rule`.

## API surface (`server.py`)

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serves `index.html` |
| `/api/entities` | GET | Returns lights, motion sensors, lux sensors, sun state |
| `/api/config` | GET/POST | Read/write `/data/adaptivelight.json`; POST also triggers threshold automation once |
| `/api/automation/status` | GET | Returns last automation run info + WS connection state |
| `/api/automation/run` | POST | Manually triggers threshold automation once (REST-based) |
| `/api/rl/status` | GET | Returns RL agent info (epsilon, steps, memory, last delta/reward) + current brightness + calibration state |
| `/api/rl/reset` | POST | Resets model weights, clears replay buffer, deletes `/data/rl_model.json` |
| `/api/rl/calibration` | GET | Returns calibration status (running/points/error) + saved curve |
| `/api/rl/calibrate` | POST | Starts calibration sweep in background thread (409 if already running) |
| `/api/rl/target` | POST | Sets `rl_target_lux`, flushes buffer, and immediately applies calibration curve — intended for external agents |

## Local development

There is no test suite and no linter config. To run the backend locally you need a real or mocked HA Supervisor environment (the `HA_TOKEN` env var and the `http://supervisor/...` URLs). The practical dev loop is:

1. Edit files.
2. Build and push the Docker image, then reload the add-on in HA — or use HA's **local add-on** feature pointing at the repo directory.

To build the Docker image manually (substitute arch as needed):

```bash
docker build \
  --build-arg BUILD_FROM=ghcr.io/home-assistant/amd64-base-python:3.11-alpine3.18 \
  -t adaptivelight-dev \
  adaptivelight/
```

## Key constraints

- The app runs inside the HA add-on sandbox; external network access goes through the Supervisor proxy URLs, not direct HA IP.
- `ingress: true` in `config.yaml` means HA proxies the UI — the app must honour the `X-Ingress-Path` header if relative paths matter.
- Config keys are strictly validated against `DEFAULT_CONFIG` in `server.py`; unknown keys sent via POST `/api/config` are silently dropped.
- The WS automation thread reconnects every 10 s on failure and shares state with Flask routes via `_status_lock`.
