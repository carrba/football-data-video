#!/usr/bin/env bash

# Runs football-possession fully detached from the current terminal/SSH session,
# so a dropped connection can't kill it mid-run (previously this left the
# annotated video with a corrupt/missing moov atom -- see PossessionPipeline).
#
# Usage: scripts/run_detached.sh --video path/to/match.mp4 --config config/default.yaml [...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/football-possession_$(date +%Y%m%d_%H%M%S).log"

echo "Logging to $LOG_FILE"

setsid nohup football-possession "$@" >"$LOG_FILE" 2>&1 </dev/null &
PID=$!
disown

echo "Started detached with PID $PID"
echo "Tail progress with: tail -f $LOG_FILE"
echo "Check completion with: test -f <output-dir>/_SUCCESS && echo done"
