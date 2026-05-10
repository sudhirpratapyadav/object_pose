#!/usr/bin/env bash
# Graceful shutdown of the object_pose stack.
#
# Sends SIGINT to the server so the OSC subprocess parks the arm at home
# in position-control mode before exiting. Then kills vite. Up to 35 s.
#
# Usage:
#   scripts/stop.sh               # graceful (SIGINT → wait → vite kill)
#   scripts/stop.sh --force       # SIGKILL everything immediately

set -e

PIDFILE_SERVER="/tmp/object_pose_server.pid"
PIDFILE_VITE="/tmp/object_pose_vite.pid"
SERVER_LOG="/tmp/object_pose_server.log"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

stop_pidfile() {
    local pidfile="$1"
    local label="$2"
    local sig="$3"
    if [[ ! -f "$pidfile" ]]; then
        # Fall back to pkill in case the pidfile is missing.
        if pgrep -f "$label" >/dev/null 2>&1; then
            echo "[stop] $label has no pidfile but is running — pkill -$sig"
            pkill -"$sig" -f "$label" 2>/dev/null || true
        fi
        return
    fi
    local pid
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
        echo "[stop] sending SIG$sig to $label (PID $pid)…"
        kill -"$sig" "$pid" 2>/dev/null || true
    else
        echo "[stop] $label PID $pid not running (stale pidfile)"
    fi
}

# ── Server (graceful) ─────────────────────────────────────────────────────────
if [[ "$FORCE" -eq 0 ]]; then
    if [[ -f "$PIDFILE_SERVER" ]] && kill -0 "$(cat "$PIDFILE_SERVER")" 2>/dev/null; then
        SERVER_PID=$(cat "$PIDFILE_SERVER")
        echo "[stop] graceful SIGINT to web_server.py (PID $SERVER_PID)…"
        echo "[stop] (waits up to 35 s for the OSC subprocess to park the arm)"
        kill -INT "$SERVER_PID" 2>/dev/null || true
        # Wait for the parent to exit. SST + JointMove home can take ~15-20s.
        for _ in $(seq 1 35); do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[stop] ⚠ server didn't exit in 35 s; force-killing."
            pkill -9 -f web_server.py 2>/dev/null || true
        else
            echo "[stop] server exited cleanly."
        fi
    else
        # Maybe a server we didn't start. Try anyway.
        if pgrep -f "web_server.py" >/dev/null 2>&1; then
            echo "[stop] no server pidfile but web_server.py is running — SIGINT to all"
            pkill -INT -f web_server.py 2>/dev/null || true
            sleep 5
            pkill -9 -f web_server.py 2>/dev/null || true
        else
            echo "[stop] no server running."
        fi
    fi
else
    echo "[stop] --force: SIGKILL web_server.py"
    pkill -9 -f web_server.py 2>/dev/null || true
fi
rm -f "$PIDFILE_SERVER"

# ── Vite ──────────────────────────────────────────────────────────────────────
if [[ "$FORCE" -eq 0 ]]; then
    stop_pidfile "$PIDFILE_VITE" "vite" "INT"
    sleep 1
fi
pkill -9 -f vite 2>/dev/null || true
rm -f "$PIDFILE_VITE"

# ── Final state ───────────────────────────────────────────────────────────────
remaining=$(pgrep -af "web_server.py|vite" | grep -v "pgrep\|grep\|stop.sh" || true)
if [[ -n "$remaining" ]]; then
    echo "[stop] ⚠ still running:"
    echo "$remaining"
else
    echo "[stop] ─── all gone ───"
fi
