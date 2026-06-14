# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Home Assistant add-on that displays a dashboard of lighting-relevant entities (lights, motion sensors, lux sensors, sun data) and provides basic lux-based automation. It is installed via the HA add-on store by pointing to this repo URL.

## Architecture

The add-on is a single Python/Flask app with no database and no build step.

- **`adaptivelight/app/server.py`** — Flask backend. Calls the HA Supervisor REST API (`http://supervisor/core/api`) for entity states and uses a persistent WebSocket connection (`ws://supervisor/core/websocket`) in a background thread for real-time lux automation. Config is stored in `/data/adaptivelight.json` (inside the container).
- **`adaptivelight/app/templates/index.html`** — Entire frontend: HTML + CSS + vanilla JS in one file. Polls `/api/entities` every 30 s. Has tabs for Sun, Lights, Motion, Lux, and an Automation settings panel.
- **`adaptivelight/run.sh`** — Container entrypoint (bashio). Sets `HA_TOKEN` from `$SUPERVISOR_TOKEN`, then starts `server.py`.
- **`adaptivelight/config.yaml`** — Add-on manifest. Declares ingress on port 8099, requires both `hassio_api` and `homeassistant_api`.
- **`adaptivelight/build.yaml`** — Multi-arch base images (Python 3.11 / Alpine 3.18).
- **`repository.yaml`** — Marks this repo as an HA add-on repository.

## API surface (`server.py`)

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serves `index.html` |
| `/api/entities` | GET | Returns lights, motion sensors, lux sensors, sun state |
| `/api/config` | GET/POST | Read/write `/data/adaptivelight.json`; POST also triggers automation once |
| `/api/automation/status` | GET | Returns last automation run info + WS connection state |
| `/api/automation/run` | POST | Manually triggers automation once (REST-based) |

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
