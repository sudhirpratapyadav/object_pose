"""Standalone RealSense diagnostic.

Run from the repo root:

    .venv/bin/python test/test_realsense.py

Reports:
- Detected devices (name + serial + firmware + USB type)
- For each, opens RGB + depth streams at 640x480x30, captures for ~3s,
  reports frames-per-second + first/last RGB pixel values + depth stats.
- Optionally writes the first valid RGB frame and depth (as PNG / NPY) to
  test/_realsense_dump/ for visual inspection.

Use this when the live pipeline reports rgb_fps=0 to isolate whether the
problem is the camera or our code path.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyrealsense2 as rs


REPO_ROOT = Path(__file__).resolve().parent.parent
DUMP_DIR  = Path(__file__).resolve().parent / "_realsense_dump"


def list_devices() -> list[rs.device]:
    ctx = rs.context()
    devs = list(ctx.devices)
    if not devs:
        print("No RealSense devices found.")
        print("Check: USB cable, lsusb | grep Intel, dmesg for permission errors.")
        return []
    print(f"Found {len(devs)} RealSense device(s):")
    for i, d in enumerate(devs):
        try:
            name   = d.get_info(rs.camera_info.name)
            serial = d.get_info(rs.camera_info.serial_number)
            fw     = d.get_info(rs.camera_info.firmware_version)
            usb    = d.get_info(rs.camera_info.usb_type_descriptor)
        except Exception as exc:
            print(f"  [{i}] info read failed: {exc}")
            continue
        print(f"  [{i}] {name}  serial={serial}  fw={fw}  usb={usb}")
    return devs


def capture(device: rs.device, *, width: int, height: int, fps: int,
            duration_s: float, want_depth: bool, do_reset: bool,
            dump_first: bool) -> None:
    serial = device.get_info(rs.camera_info.serial_number)
    print(f"\n=== Testing {device.get_info(rs.camera_info.name)} ({serial}) ===")

    if do_reset:
        print("  hw_resetting...")
        device.hardware_reset()
        time.sleep(2.0)
        # Re-resolve the device after reset (its handle may be stale).
        for d in rs.context().devices:
            if d.get_info(rs.camera_info.serial_number) == serial:
                device = d
                break

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    if want_depth:
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    print(f"  starting pipeline ({width}x{height}@{fps} color"
          f"{', depth' if want_depth else ''})...")
    try:
        profile = pipe.start(cfg)
    except RuntimeError as exc:
        print(f"  pipe.start failed: {exc}")
        return

    try:
        intr = (profile.get_stream(rs.stream.color)
                .as_video_stream_profile().get_intrinsics())
        print(f"  factory color intrinsics: "
              f"fx={intr.fx:.2f} fy={intr.fy:.2f} cx={intr.ppx:.2f} cy={intr.ppy:.2f}")
        if want_depth:
            depth_sensor = profile.get_device().first_depth_sensor()
            scale = depth_sensor.get_depth_scale()
            print(f"  depth_scale = {scale:g} m/unit")

        # Capture loop
        deadline   = time.time() + duration_s
        n_color    = 0
        n_depth    = 0
        first_rgb  = None
        last_rgb   = None
        first_dpt  = None
        first_log  = True
        timeouts   = 0

        while time.time() < deadline:
            try:
                frames = pipe.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                timeouts += 1
                continue
            color = frames.get_color_frame()
            if color:
                arr = np.asanyarray(color.get_data())
                if first_rgb is None:
                    first_rgb = arr.copy()
                last_rgb = arr
                n_color += 1
            if want_depth:
                d = frames.get_depth_frame()
                if d:
                    darr = np.asanyarray(d.get_data())
                    if first_dpt is None:
                        first_dpt = darr.copy()
                    n_depth += 1
            if first_log and (n_color > 0 or n_depth > 0):
                first_log = False
                print(f"  first frame received at "
                      f"+{time.time() - (deadline - duration_s):.2f}s "
                      f"(color={n_color}, depth={n_depth})")

        elapsed = duration_s
        print(f"  result over {elapsed:.1f}s:")
        print(f"    color frames: {n_color}  ({n_color / elapsed:.1f} fps)")
        if want_depth:
            print(f"    depth frames: {n_depth}  ({n_depth / elapsed:.1f} fps)")
        print(f"    timeouts: {timeouts}")

        if first_rgb is not None:
            print(f"    first rgb pixel (0,0)  = {first_rgb[0, 0]}")
            print(f"    first rgb mean         = {first_rgb.mean():.1f}")
            print(f"    first rgb stdev        = {first_rgb.std():.1f}")
            if last_rgb is not None and last_rgb is not first_rgb:
                # If pixels are *identical* between first and last, the
                # frame is being repeated (sensor stuck).
                diff = float(np.mean(np.abs(last_rgb.astype(int)
                                             - first_rgb.astype(int))))
                print(f"    mean abs diff first vs last rgb: {diff:.1f}")
                if diff < 0.01 and n_color > 1:
                    print("    WARNING: identical first and last frame "
                          "— sensor may be stuck / repeating one frame")

        if want_depth and first_dpt is not None:
            valid = first_dpt[first_dpt > 0]
            if valid.size:
                m_min = valid.min() * (depth_sensor.get_depth_scale())
                m_max = valid.max() * (depth_sensor.get_depth_scale())
                m_med = float(np.median(valid)) * depth_sensor.get_depth_scale()
                print(f"    first depth: valid_pix={valid.size}/"
                      f"{first_dpt.size} ({100 * valid.size / first_dpt.size:.0f}%)  "
                      f"min={m_min:.3f}m  med={m_med:.3f}m  max={m_max:.3f}m")
            else:
                print("    first depth: ALL ZERO — depth sensor not producing data")

        if dump_first and (first_rgb is not None or first_dpt is not None):
            DUMP_DIR.mkdir(exist_ok=True)
            if first_rgb is not None:
                import cv2
                bgr = cv2.cvtColor(first_rgb, cv2.COLOR_RGB2BGR)
                p = DUMP_DIR / f"first_rgb_{serial}.png"
                cv2.imwrite(str(p), bgr)
                print(f"    wrote {p}")
            if first_dpt is not None:
                p = DUMP_DIR / f"first_depth_{serial}.npy"
                np.save(str(p), first_dpt)
                print(f"    wrote {p}")

        if n_color == 0:
            print("\n  ❌ NO COLOR FRAMES RECEIVED.")
            print("     Things to check:")
            print("       • USB cable: USB-C ↔ USB-A or USB-C ↔ USB-C, must be "
                  "USB 3 capable. RealSense on USB 2 will not deliver "
                  f"{width}x{height}@{fps}.")
            print("       • lsusb -t  (look for 5000M speed = USB3)")
            print("       • Try a different USB port (preferably USB-C 3.x).")
            print("       • Try `realsense-viewer` from the Intel SDK to verify "
                  "the camera works at OS level outside Python.")
        elif n_color < fps * elapsed * 0.5:
            print(f"\n  ⚠  Got {n_color/elapsed:.1f} fps but expected ~{fps}. "
                  "USB bandwidth or driver issue.")
        else:
            print(f"\n  ✓ Color stream healthy at {n_color/elapsed:.1f} fps.")
    finally:
        try:
            pipe.stop()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width",  type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps",    type=int, default=30)
    ap.add_argument("--duration", type=float, default=3.0,
                    help="Capture duration in seconds.")
    ap.add_argument("--no-depth", action="store_true",
                    help="Skip the depth stream (RGB only).")
    ap.add_argument("--reset", action="store_true",
                    help="Hardware-reset the device before opening it.")
    ap.add_argument("--dump", action="store_true",
                    help="Write the first valid RGB+depth frames to "
                         "test/_realsense_dump/.")
    args = ap.parse_args()

    devs = list_devices()
    if not devs:
        sys.exit(1)
    for d in devs:
        capture(d,
                width=args.width, height=args.height, fps=args.fps,
                duration_s=args.duration,
                want_depth=not args.no_depth,
                do_reset=args.reset,
                dump_first=args.dump)


if __name__ == "__main__":
    main()
