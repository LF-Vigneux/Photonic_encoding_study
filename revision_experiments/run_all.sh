#!/bin/bash
# Usage (from Photonic_encoding_study/, venv active):
#   bash revision_experiments/run_all.sh smoke      # ~minutes, checks the code runs
#   bash revision_experiments/run_all.sh full       # overnight; a failing step does not stop the others
# Logs: results/revisions_logs/<step>.log
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD/revision_experiments:$PWD"
mkdir -p results/revisions_logs
python -c "import numpy, torch, merlin, perceval, sklearn, scipy, pandas" 2>&1 || {
  echo "Missing packages. Active python: $(which python). Activate the venv and run pip install -r requirements.txt."; exit 1; }
# RUNPREFIX (optional) wraps each step, e.g. a systemd memory cap (see night.sh)
run() { name=$1; shift; echo "[$(date +%T)] start $name"
        $RUNPREFIX python -u revision_experiments/"$@" > results/revisions_logs/$name.log 2>&1; rc=$?
        if [ $rc -eq 0 ]; then echo "[$(date +%T)] OK    $name"
        else echo "[$(date +%T)] FAIL  $name rc=$rc$([ $rc -eq 137 ] && echo ' (SIGKILL, probably out of memory)') (see results/revisions_logs/$name.log)"; fi; }
if [ "$1" = smoke ]; then
  run smoke_resources   exp_resources.py --config resources.json
  run smoke_headroom    exp_headroom_svm.py --config smoke.json
  run smoke_controls    exp_controls.py --config controls_smoke.json
else
  run resources         exp_resources.py --config resources.json
  run headroom_svm      exp_headroom_svm.py --config headroom_svm.json
  run controls          exp_controls.py --config controls.json
  run statistics_main   exp_statistics.py --config statistics_main.json
  # statistics_heldout.json is NOT run here: freeze its prereg block first.
fi
