#!/usr/bin/with-contenv bashio

bashio::log.info "Starting AdaptiveLight Monitor..."

export HA_TOKEN="${SUPERVISOR_TOKEN}"

python3 /app/server.py
