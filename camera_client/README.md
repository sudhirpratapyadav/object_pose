# camera_client

Run this on the PC that has the camera attached. It serves the RGB stream as
MJPEG over HTTP plus a small JSON `/intrinsics` endpoint. The depth/viewer PC
connects via `camera.NetworkRGB("http://<this-pc>:8080")`.

## Install

Only two deps needed on the camera PC:

```bash
pip install opencv-python numpy
# plus pyrealsense2 if --source realsense
pip install pyrealsense2
```

## Run

RealSense:

```bash
python server.py --source realsense --width 1280 --height 720 --fps 30
```

Generic webcam (OpenCV):

```bash
python server.py --source 0 --width 1280 --height 720 --fps 30 \
    --fx 900 --fy 900 --cx 640 --cy 360
```

Open `http://<this-pc>:8080/` in a browser to verify before connecting from
the depth PC.

## Notes

- For a generic webcam, intrinsics are a placeholder (defaults to fx=fy=width,
  centered). Calibrate your camera and pass `--fx/--fy/--cx/--cy` for accurate
  metric depth.
- The HTTP server uses Python's stdlib only; OpenCV is used for JPEG encoding
  and to read non-RealSense sources.
