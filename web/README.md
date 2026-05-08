# Object Pose — Custom Web Viewer (React + Three.js)

Vite + React + Three.js frontend that streams from `web_server.py` over a binary
WebSocket. Point cloud uses a custom `ShaderMaterial` so per-point screen size
can grow with depth (far-bigger effect that viser can't do).

## Run

Backend:
```
uv run python web_server.py
# defaults: ws://0.0.0.0:8765, RealSense camera, default depth model
```

Frontend (one-time):
```
cd web
npm install
npm run dev
# open http://localhost:5173
```

## Wire format

Little-endian binary frames:

```
HEADER : 'P3DF' u32 | seq u32 | kind u8 | _pad u24
kind=0 points : n u32 | xyz_f16 [3n] | rgb_u8 [3n]
kind=1 mesh   : nv u32 | nf u32 | xyz_f16 [3nv] | rgb_u8 [3nv] | faces_u32 [3nf]
kind=2 jpeg   : w u16 | h u16 | jpeg bytes
kind=3 meta   : utf-8 json
```

Float16 positions cut bandwidth in half vs float32 with no visible quality loss
on metric depth in [0, 10] m.

## Why not viser

- Per-vertex point sizing not exposed (we want far-bigger).
- This stack also sets you up to plug in WebGPU/WGSL shaders, custom culling,
  and time-scrubbing later.
