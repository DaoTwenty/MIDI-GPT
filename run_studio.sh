#!/bin/bash
export PYTHONPATH=src/python
.venv/bin/python -m midigpt.osc.server --ckpt models/yellow.pt > tmp/osc_server.log 2>&1 &
.venv/bin/python -m midigpt.osc.studio.app --port 5001 --log-level DEBUG > tmp/studio_ui.log 2>&1 &
wait
