# test/

Standalone diagnostics. Each script runs in isolation against a single
subsystem (camera, robot, etc.) so you can identify which part of the
pipeline is broken without dragging in the rest of the server.

## test_realsense.py

Dumps RealSense diagnostics to stdout. Use when the live pipeline reports
`rgb 0.0 fps` to figure out whether the camera is at fault or our code is.

```bash
.venv/bin/python test/test_realsense.py                   # default 640x480x30, 3s
.venv/bin/python test/test_realsense.py --reset           # hw_reset before opening
.venv/bin/python test/test_realsense.py --dump            # save first frame
.venv/bin/python test/test_realsense.py --duration 10     # longer capture
.venv/bin/python test/test_realsense.py --no-depth        # color only
```

Reports per-device: name / serial / firmware / USB-type, factory intrinsics,
depth scale, frames-per-second, first/last RGB pixel sanity, depth coverage.
Writes diagnostic dumps to `test/_realsense_dump/` when `--dump` is given.

## Adding more tests

Anything that needs to talk to a single hardware component goes here.
Future candidates: `test_kinova.py` (connect-and-read-state, no torque
mode), `test_mjcf.py` (load + render a frame), `test_calib_yaml.py`
(round-trip the YAML loaders).

Tests are not part of the wheel; they're meant to be run from the repo
root. Each script handles its own argparse so you don't need pytest.
