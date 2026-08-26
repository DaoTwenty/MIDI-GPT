#!/bin/bash

# MIDI-GPT Studio Startup Script
# This script launches the OSC server and the Studio UI in the background.

# 1. Setup paths
PROJECT_ROOT="$(pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src/python"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

echo "🚀 Starting MIDI-GPT Studio..."

# 2. Cleanup existing processes to free up ports
echo "🧹 Cleaning up old processes..."
pkill -f midigpt.osc.server 2>/dev/null
pkill -f midigpt.osc.studio.app 2>/dev/null
sleep 2

# 3. Create logs directory if it doesn't exist
mkdir -p tmp

# 4. Launch OSC Server
echo "🎹 Launching OSC Server (models/yellow.pt)..."
${VENV_PYTHON} -m midigpt.osc.server --ckpt models/yellow.pt --reply-port 7401 > tmp/osc_server.log 2>&1 &
SERVER_PID=$!

# 5. Wait for server to initialize
sleep 3

# 6. Launch Studio UI
echo "🌐 Launching Studio UI on port 5001..."
${VENV_PYTHON} -m midigpt.osc.studio.app --port 5001 --log-level DEBUG > tmp/studio_ui.log 2>&1 &
UI_PID=$!

echo "✅ Started!"
echo "   - Studio UI: http://localhost:5001"
echo "   - OSC Server PID: ${SERVER_PID}"
echo "   - Studio UI PID: ${UI_PID}"
echo ""
echo "Monitor logs with:"
echo "   tail -f tmp/osc_server.log"
echo "   tail -f tmp/studio_ui.log"
echo ""
echo "Press Ctrl+C to stop this script (processes will remain running in background)."
wait
