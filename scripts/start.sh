#!/usr/bin/env bash
# Full bring-up: kortex sanity → robot recovery → camera reset → vite → server.
#
# Usage:
#   scripts/start.sh                # default: hardware mode
#   scripts/start.sh --mode sim     # sim mode (skips robot recovery)
#   scripts/start.sh --no-vite      # skip starting vite (e.g. it's already up)
#
# Logs:
#   /tmp/object_pose_server.log
#   /tmp/object_pose_vite.log
#
# Stop with scripts/stop.sh.

set -e

# Resolve repo root from this script's location, regardless of where it's
# invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
ROBOT_IP="${ROBOT_IP:-192.168.1.10}"
SERVER_LOG="/tmp/object_pose_server.log"
VITE_LOG="/tmp/object_pose_vite.log"
PIDFILE_SERVER="/tmp/object_pose_server.pid"
PIDFILE_VITE="/tmp/object_pose_vite.pid"

# ── Parse args ────────────────────────────────────────────────────────────────
MODE="real"
ROBOT_SOURCE=""   # auto-resolve: hardware when --mode real, sim when --mode sim
START_VITE=1
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"; shift 2 ;;
        --robot-source)
            ROBOT_SOURCE="$2"; shift 2 ;;
        --no-vite)
            START_VITE=0; shift ;;
        --no-robot)
            ROBOT_SOURCE="none"; shift ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Auto-resolve ROBOT_SOURCE based on --mode if the user didn't set it.
if [[ -z "$ROBOT_SOURCE" ]]; then
    if [[ "$MODE" == "sim" ]]; then
        ROBOT_SOURCE="sim"
    else
        ROBOT_SOURCE="hardware"
    fi
fi
# Forwarded EXTRA_ARGS land at the end of the python command.

echo "─────────────────────────────────────────"
echo "  object_pose — start"
echo "─────────────────────────────────────────"
echo "  mode:           $MODE"
echo "  robot-source:   $ROBOT_SOURCE"
[[ -n "${EXTRA_ARGS[*]:-}" ]] && echo "  extra:          ${EXTRA_ARGS[*]}"
echo "  python:         $PYTHON"
echo "  cwd:            $(pwd)"
echo

# ── Kill any existing instance first ──────────────────────────────────────────
if [[ -f "$PIDFILE_SERVER" ]] && kill -0 "$(cat "$PIDFILE_SERVER")" 2>/dev/null; then
    echo "[start] previous server still running (PID $(cat "$PIDFILE_SERVER"))."
    echo "[start] run scripts/stop.sh first, or it will be force-killed."
    pkill -9 -f "web_server.py" 2>/dev/null || true
    sleep 1
fi

# ── Hardware-mode preflight ───────────────────────────────────────────────────
if [[ "$ROBOT_SOURCE" == "hardware" && "$MODE" == "real" ]]; then
    echo "[start] hardware preflight…"

    # 1) ping
    if ! ping -c 1 -W 2 "$ROBOT_IP" >/dev/null 2>&1; then
        echo "[start] ❌ no ping to $ROBOT_IP. Is the robot powered + on the network?"
        exit 1
    fi
    echo "[start] ping ok"

    # 2) wait for kortex TCP
    until "$PYTHON" -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(2)
try: s.connect(('$ROBOT_IP', 10000)); s.close(); exit(0)
except Exception: exit(1)
" 2>/dev/null; do
        echo "[start] waiting for kortex on $ROBOT_IP:10000…"
        sleep 5
    done
    echo "[start] kortex ok"

    # 3) clear any leftover fault / low-level mode from a prior session
    "$PYTHON" -c "
from hardware import KinovaHardware
import time
hw = KinovaHardware('$ROBOT_IP')
hw.connect()
print(f'[start] before: state={hw.base.GetArmState().active_state} mode={hw.get_servoing_mode()}')
if hw.get_servoing_mode() == 'LOW_LEVEL':
    hw.set_servoing_mode(low_level=False); time.sleep(0.5)
hw.clear_faults(); time.sleep(2.0)
print(f'[start] after:  state={hw.base.GetArmState().active_state} mode={hw.get_servoing_mode()}')
hw.disconnect()
"
fi

# ── RealSense reset (only in real mode) ──────────────────────────────────────
if [[ "$MODE" == "real" ]]; then
    echo "[start] resetting RealSense…"
    "$PYTHON" -c "
import pyrealsense2 as rs, time
devs = list(rs.context().devices)
if not devs:
    print('[start] (no RealSense detected; skipping)')
else:
    for d in devs:
        d.hardware_reset()
    time.sleep(3)
    print('[start] camera reset ok')
" || echo "[start] (camera reset failed — continuing anyway)"
fi

# ── Vite (web frontend dev server) ────────────────────────────────────────────
if [[ "$START_VITE" -eq 1 ]]; then
    if [[ -f "$PIDFILE_VITE" ]] && kill -0 "$(cat "$PIDFILE_VITE")" 2>/dev/null; then
        echo "[start] vite already running (PID $(cat "$PIDFILE_VITE"))."
    else
        echo "[start] launching vite…"
        cd "$REPO_ROOT/web"
        nohup npm run dev > "$VITE_LOG" 2>&1 &
        echo $! > "$PIDFILE_VITE"
        cd "$REPO_ROOT"
        # Give vite a moment to bind 5173.
        for _ in $(seq 1 10); do
            if grep -q "Local:" "$VITE_LOG" 2>/dev/null; then break; fi
            sleep 0.5
        done
        echo "[start] vite at http://localhost:5173/  (log: $VITE_LOG)"
    fi
fi

# ── Server ────────────────────────────────────────────────────────────────────
echo "[start] launching web_server.py…"
SERVER_CMD=(
    "$PYTHON" "$REPO_ROOT/web_server.py"
    --mjcf "$REPO_ROOT/robot/mjcf/scene.xml"
    --robot-source "$ROBOT_SOURCE"
    --robot-ip "$ROBOT_IP"
    --mode "$MODE"
    "${EXTRA_ARGS[@]}"
)
nohup "${SERVER_CMD[@]}" > "$SERVER_LOG" 2>&1 &
echo $! > "$PIDFILE_SERVER"

# Wait for boot (look for `[ws] listening`).
echo "[start] waiting for server to boot…"
for _ in $(seq 1 30); do
    if grep -q "\[ws\] listening" "$SERVER_LOG" 2>/dev/null; then
        break
    fi
    if grep -qE "Traceback|Calibration config not found|aborting" "$SERVER_LOG" 2>/dev/null; then
        echo
        echo "[start] ❌ server failed to boot. Last lines:"
        tail -30 "$SERVER_LOG"
        exit 1
    fi
    sleep 1
done

echo "[start] ─── ready ───"
echo "  PID:    $(cat "$PIDFILE_SERVER")"
echo "  log:    $SERVER_LOG"
echo "  ui:     http://localhost:5173/"
echo
echo "  follow logs:   tail -f $SERVER_LOG"
echo "  shutdown:      scripts/stop.sh"
