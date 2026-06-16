# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Home Assistant add-on that displays a dashboard of lighting-relevant entities (lights, motion sensors, lux sensors, sun data) and provides lux-based automation. Installed via the HA add-on store by pointing to this repo URL.

## Architecture

Single Python/Flask app — no database, no build step.

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

**Neural network:** 5 → 32 → 16 → 11 (ReLU hidden, linear output, He init, manual SGD backprop — pure numpy, no framework).

**State vector:** `[lux_error_norm, brightness_norm, lux_trend_norm, prev_action_norm, goal_norm]`
- `lux_error_norm` = `(current_lux − target_lux) / target_lux`, clipped to [−3, 3]
- `brightness_norm` = current RL-tracked brightness / 100
- `lux_trend_norm` = change since last reading / target_lux, clipped to [−1, 1]
- `prev_action_norm` = last brightness delta / 25
- `goal_norm` = `target_lux / 1000.0` — embeds the lux goal so one model serves all targets (multi-goal DQN)

**Actions (11 discrete):** brightness delta in % — `[−25, −15, −10, −5, −2, 0, +2, +5, +10, +15, +25]`

**Reward:** `+2.0` within ±`rl_lux_tolerance` lux of target; otherwise `−|error_pct| × 2`, bonus `+0.5` at <15 % error, `+0.4` if improving vs. previous step. Clipped to [−3, 3].

**Training:** experience replay (buffer 4 000, batch 8), target network synced every 50 steps, ε-greedy (0.60 → 0.05, decay 0.99/step). After simulation, `set_live_mode()` fixes ε=0.05 and switches to `lr_live=0.02` for online fine-tuning. Model auto-saved to `/data/rl_model.json` every 50 steps.

**Simulation training (`simulate()`):** Offline pre-training on a `DigitalTwin` of the calibration curve before live deployment. 40 % of episodes include a persistent ambient lux offset (positive or negative) to teach the agent that the calibration curve is a starting point, not ground truth — i.e. it must keep adjusting based on actual lux error even when the lamp is already at the "calibrated" brightness. 25 % of non-ambient episodes start from a random brightness. 5 % of individual steps inject extreme disturbances (lux=0 or lux=2×max). Triggered automatically after calibration completes, or manually via `/api/rl/simulate`.

**Policy guard (`act_guarded()`):** When `|lux − target| / target > 0.5`, restricts the action set to the correct direction only (increase-only or decrease-only). Inside ±50 % band, normal ε-greedy applies.

**Gate conditions in `_apply_rl()`:**
- RL fires only when `rl_enabled` is true and calibration is not running
- Cooldown (`rl_action_cooldown`, default 3 s) between successive adjustments
- If `rl_night_only` is true, requires sun elevation < `rl_sun_threshold` (degrees); `_sun_elevation` defaults to 90° (day) until first `sun.sun` WS event
- **Spike filter:** lux changes > `max(target × 0.7, 20)` in one step are skipped (sensor covered/uncovered); `prev_lux` advances so the next event starts cleanly
- **Dead band:** if `|lux − target| ≤ rl_lux_tolerance`, the command is skipped but the agent still learns from the transition
- **Stuck-at-boundary detection:** if the agent is at minimum brightness (≤5.5 %) while lux remains >10 % below target for 6 consecutive steps, a warm-start is forced

**Warm-start:** On the first RL step per session (`prev_state is None`), the lamp is immediately commanded to `calib.brightness_for_lux(target)` and the agent returns without acting — the following lux event is the actual first decision. Changing `rl_target_lux` flushes the replay buffer and triggers another warm-start.

**Threading:** `RLAgent` is a singleton (`_get_rl_agent()`). Its `_lock` guards network weights. Brightness is tracked in module-level `_rl_brightness` guarded by `_rl_lock`. A sensor in `rl_input_sensors` is never passed to the threshold path (`_apply_rule`).

## API surface (`server.py`)

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serves `index.html` |
| `/api/entities` | GET | Returns lights, motion sensors, lux sensors, sun state |
| `/api/config` | GET/POST | Read/write `/data/adaptivelight.json`; POST also triggers threshold automation once |
| `/api/automation/status` | GET | Returns last automation run info + WS connection state |
| `/api/automation/run` | POST | Manually triggers threshold automation once (REST-based) |
| `/api/rl/status` | GET | Returns RL agent info (epsilon, steps, memory, last delta/reward) + current brightness + calibration/simulation state |
| `/api/rl/reset` | POST | Resets model weights, clears replay buffer, deletes `/data/rl_model.json` |
| `/api/rl/calibration` | GET | Returns calibration status (running/points/error) + saved curve |
| `/api/rl/calibrate` | POST | Starts calibration sweep in background thread (409 if already running); auto-triggers simulation training on completion |
| `/api/rl/simulate` | POST | Starts simulation training in background thread (409 if already running) |
| `/api/rl/simulate/status` | GET | Returns simulation progress (episode, total, epsilon, done) |
| `/api/rl/target` | POST | Sets `rl_target_lux`, flushes buffer, and immediately applies calibration curve — intended for external agents |
| `/api/rl/export` | GET | Returns model weights + calibration data for export |

## Config keys (`DEFAULT_CONFIG` in `server.py`)

Unknown keys sent via POST `/api/config` are silently dropped. RL-specific keys:

| Key | Default | Purpose |
|---|---|---|
| `rl_target_lux` | 100 | Target illuminance in lux |
| `rl_action_cooldown` | 3.0 | Minimum seconds between RL brightness adjustments |
| `rl_lux_tolerance` | 0.5 | Dead band ±lux; no command issued inside |
| `rl_night_only` | false | Restrict RL to night (sun elevation gate) |
| `rl_sun_threshold` | 0.0 | Sun elevation threshold in degrees for night gate |
| `rl_sim_episodes` | 1000 | Episodes for simulation training |
| `rl_sim_steps_per_ep` | 30 | Steps per simulation episode |
| `rl_sim_noise_std` | 2.0 | Gaussian lux noise std dev in digital twin |
| `rl_sim_goals` | [] | Explicit goal lux values for simulation; empty = auto from calibration range |

## Local development

No test suite, no linter config. Requires a real or mocked HA Supervisor environment (`HA_TOKEN` env var, `http://supervisor/...` URLs). Practical dev loop:

1. Edit files.
2. Build and push the Docker image, then reload the add-on in HA — or use HA's **local add-on** feature pointing at the repo directory.

```bash
docker build \
  --build-arg BUILD_FROM=ghcr.io/home-assistant/amd64-base-python:3.11-alpine3.18 \
  -t adaptivelight-dev \
  adaptivelight/
```

## Key constraints

- The app runs inside the HA add-on sandbox; external network access goes through the Supervisor proxy URLs, not direct HA IP.
- `ingress: true` in `config.yaml` means HA proxies the UI — the app must honour the `X-Ingress-Path` header if relative paths matter.
- The WS automation thread reconnects every 10 s on failure and shares state with Flask routes via `_status_lock`.
- Persistent data (`/data/adaptivelight.json`, `/data/rl_model.json`, `/data/rl_calibration.json`) survives add-on restarts but lives inside the container volume — not in the repo.
- Changing `STATE_SIZE` in `rl_agent.py` breaks saved model compatibility; `load()` checks `net.sizes[0] != STATE_SIZE` and discards incompatible checkpoints silently.
