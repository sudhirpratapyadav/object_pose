#!/usr/bin/env bash
# Bring up the camera + UI without the robot. Real RealSense, no kortex
# preflight. Useful for testing depth backends (camera-depth, MoGe,
# DAv2, FoundationStereo, ...) end-to-end while the arm is powered off.
#
# Usage:
#   scripts/start_no_robot.sh              # real camera, no robot
#   scripts/start_no_robot.sh --no-vite    # skip starting vite
#
# Stop with scripts/stop.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/start.sh" --no-robot --mode real "$@"
