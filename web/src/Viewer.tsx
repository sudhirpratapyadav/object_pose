import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { RobotRenderer } from "./RobotRenderer";
import { StreamState } from "./useStream";

type Props = {
  stream: StreamState;
  pointSize: number;
  inversePerspective: boolean;
  display: "points" | "mesh" | "both";
  showCamera: boolean;
  showBBox: boolean;
  showRobot: boolean;
  showEeAxes: boolean;
  showWorldAxes: boolean;
  showWorldHandle: boolean;
  gizmoMode: "translate" | "rotate";
};

const POINT_VS = /* glsl */ `
attribute vec3 color;
attribute float mask;
varying vec3 vColor;
varying float vMask;
uniform float uBaseSize;
uniform float uPxPerMeter;
uniform float uInversePerspective;
void main() {
  vColor = color;
  vMask = mask;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  float depth = max(-mv.z, 0.001);
  gl_Position = projectionMatrix * mv;
  float inv = uInversePerspective;
  gl_PointSize = uBaseSize * uPxPerMeter * mix(1.0 / depth, 1.0, inv);
}`;

const POINT_FS = /* glsl */ `
varying vec3 vColor;
varying float vMask;
uniform vec3 uHighlight;        // tint color for masked points
uniform float uHighlightAmt;    // 0..1 mix
void main() {
  vec2 p = gl_PointCoord - 0.5;
  if (dot(p, p) > 0.25) discard;
  vec3 c = mix(vColor, uHighlight, vMask * uHighlightAmt);
  gl_FragColor = vec4(c, 1.0);
}`;

export function Viewer({ stream, pointSize, inversePerspective, display, showCamera, showBBox, showRobot, showEeAxes, showWorldAxes, showWorldHandle, gizmoMode }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const propsRef = useRef({ pointSize, inversePerspective, display, showCamera, showBBox, showRobot, showEeAxes, showWorldAxes, showWorldHandle, gizmoMode });
  propsRef.current = { pointSize, inversePerspective, display, showCamera, showBBox, showRobot, showEeAxes, showWorldAxes, showWorldHandle, gizmoMode };
  // Mirror `stream` into a ref so the (mount-only) effect always sees the
  // latest meta + state objects, not the ones captured at mount.
  const streamRef = useRef(stream);
  streamRef.current = stream;

  useEffect(() => {
    const container = containerRef.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setClearColor(0x0b0d10);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.01, 100);
    // World is Z-up (MuJoCo); look at origin from above-and-to-the-side.
    camera.position.set(1.5, -1.5, 1.0);
    camera.up.set(0, 0, 1);
    camera.lookAt(0, 0, 0);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.update();

    // World axes at origin (X red, Y green, Z blue).
    const worldAxes = new THREE.AxesHelper(0.20);
    worldAxes.name = "world-axes";
    scene.add(worldAxes);

    // Lights for the MJCF robot (MeshLambertMaterial). Other point/mesh
    // surfaces use ShaderMaterials and are unaffected.
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambient);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.7);
    dirLight.position.set(0.5, -1.0, -0.5);
    scene.add(dirLight);

    // Robot (optional — populated when KIND_ROBOT_GEOMETRY arrives).
    const robot = new RobotRenderer();
    scene.add(robot.root);
    let lastRobotGeomSeq = -1, lastRobotXformSeq = -1;

    // EE axes (live): independent of the robot mesh — drawn at
    // pinch_site (point between gripper fingers), driven by
    // controller_state.ee_pose each frame. Shown even when the robot
    // mesh is hidden.
    const eeLiveAxes = new THREE.AxesHelper(0.12);
    eeLiveAxes.name = "ee-live-axes";
    eeLiveAxes.visible = false;
    scene.add(eeLiveAxes);

    // Bounding box for the segmented region.
    // Bbox lines: build from raw vertices each time mask updates. Box3Helper's
    // geometry doesn't refresh when the underlying Box3 mutates, so we own the
    // geometry directly.
    const bboxGeom = new THREE.BufferGeometry();
    bboxGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(24 * 3), 3));
    const bboxMat = new THREE.LineBasicMaterial({ color: 0xff8833 });
    const bboxLines = new THREE.LineSegments(bboxGeom, bboxMat);
    bboxLines.visible = false;
    scene.add(bboxLines);

    const setBBox = (mn: ArrayLike<number>, mx: ArrayLike<number>) => {
      const [x0, y0, z0] = [mn[0], mn[1], mn[2]];
      const [x1, y1, z1] = [mx[0], mx[1], mx[2]];
      const c = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
      ];
      const edges = [
        0, 1, 1, 2, 2, 3, 3, 0,   // bottom
        4, 5, 5, 6, 6, 7, 7, 4,   // top
        0, 4, 1, 5, 2, 6, 3, 7,   // verticals
      ];
      const arr = (bboxGeom.getAttribute("position") as THREE.BufferAttribute).array as Float32Array;
      for (let i = 0; i < edges.length; i++) {
        const v = c[edges[i]];
        arr[i * 3 + 0] = v[0];
        arr[i * 3 + 1] = v[1];
        arr[i * 3 + 2] = v[2];
      }
      (bboxGeom.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
      bboxGeom.computeBoundingSphere();
    };
    // Camera pose = world frame of the physical camera, set from cam_calib.
    // The frustum geometry is in camera-local coordinates; we parent it
    // under cameraPose so the whole rig moves with the calibration.
    const cameraPose = new THREE.Group();
    cameraPose.name = "camera-pose";
    scene.add(cameraPose);

    // Camera-local axes (small, on the camera body itself).
    const camAxes = new THREE.AxesHelper(0.08);
    cameraPose.add(camAxes);

    // Camera frustum (placeholder; sized later when meta arrives).
    const frustumGeom = new THREE.BufferGeometry();
    const frustumMat = new THREE.LineBasicMaterial({ color: 0x33cc55 });
    const frustum = new THREE.LineSegments(frustumGeom, frustumMat);
    cameraPose.add(frustum);

    // Points
    const pointsGeom = new THREE.BufferGeometry();
    pointsGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
    pointsGeom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(0), 3));
    pointsGeom.setAttribute("mask", new THREE.BufferAttribute(new Float32Array(0), 1));
    const uniforms = {
      uBaseSize: { value: pointSize },
      uPxPerMeter: { value: container.clientHeight },
      uInversePerspective: { value: inversePerspective ? 1.0 : 0.0 },
      uHighlight: { value: new THREE.Color(0xffaa33) },
      uHighlightAmt: { value: 0.85 },
    };
    const pointsMat = new THREE.ShaderMaterial({
      uniforms,
      vertexShader: POINT_VS,
      fragmentShader: POINT_FS,
      transparent: false,
    });
    const points = new THREE.Points(pointsGeom, pointsMat);
    scene.add(points);

    // Mesh — custom ShaderMaterial:
    //   - tints masked vertices toward uHighlight (same as points)
    //   - optionally Lambert-shades using a per-vertex 'normal' attribute
    //     when the model provides normals (uUseNormals = 1.0).
    const meshGeom = new THREE.BufferGeometry();
    meshGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
    meshGeom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(0), 3));
    meshGeom.setAttribute("mask", new THREE.BufferAttribute(new Float32Array(0), 1));
    // aNormal attribute is only attached once mesh data with normals arrives.
    meshGeom.setIndex(new THREE.BufferAttribute(new Uint32Array(0), 1));
    const meshMat = new THREE.ShaderMaterial({
      uniforms: {
        uHighlight: uniforms.uHighlight,
        uHighlightAmt: uniforms.uHighlightAmt,
        uUseNormals: { value: 0.0 },
        // Light direction in camera frame (over-the-shoulder default).
        uLightDir: { value: new THREE.Vector3(0.4, -0.6, -0.7).normalize() },
        uAmbient:  { value: 0.35 },
      },
      vertexShader: /* glsl */ `
        attribute vec3 color;
        attribute float mask;
        attribute vec3 aNormal;
        varying vec3 vColor;
        varying float vMask;
        varying vec3 vNormal;
        void main() {
          vColor = color;
          vMask = mask;
          vNormal = aNormal;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        varying vec3 vColor;
        varying float vMask;
        varying vec3 vNormal;
        uniform vec3 uHighlight;
        uniform float uHighlightAmt;
        uniform float uUseNormals;
        uniform vec3 uLightDir;
        uniform float uAmbient;
        void main() {
          vec3 c = mix(vColor, uHighlight, vMask * uHighlightAmt);
          // If we have normals, Lambert-shade; else pass color through.
          float lit = 1.0;
          if (uUseNormals > 0.5) {
            vec3 n = normalize(vNormal);
            float ndotl = abs(dot(n, uLightDir));
            lit = uAmbient + (1.0 - uAmbient) * ndotl;
          }
          gl_FragColor = vec4(c * lit, 1.0);
        }`,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(meshGeom, meshMat);
    scene.add(mesh);

    // Click-feedback sphere: brief 400ms ping at the 3D hit point.
    const pingGeom = new THREE.SphereGeometry(0.012, 12, 8);
    const pingMat = new THREE.MeshBasicMaterial({
      color: 0xffaa33, transparent: true, opacity: 0.0, depthTest: false,
    });
    const ping = new THREE.Mesh(pingGeom, pingMat);
    ping.renderOrder = 999;
    ping.visible = false;
    scene.add(ping);
    let pingT0 = 0;

    // Raycaster for picking. Threshold for points hit-test scales with the
    // current point size so it feels right.
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 0.02 };

    // Track mouse-down to suppress clicks that were really drags (orbit/pan).
    let downX = 0, downY = 0;
    const CLICK_MOVE_TOL = 4;  // px
    const onCanvasMouseDown = (e: MouseEvent) => {
      downX = e.clientX;
      downY = e.clientY;
    };

    const onCanvasClick = (e: MouseEvent) => {
      if (Math.abs(e.clientX - downX) > CLICK_MOVE_TOL ||
          Math.abs(e.clientY - downY) > CLICK_MOVE_TOL) {
        return;  // it was a drag, not a click
      }
      const meta = streamRef.current.meta;
      if (!meta) return;
      const rect = renderer.domElement.getBoundingClientRect();
      const ndc = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1,
      );
      raycaster.setFromCamera(ndc, camera);
      raycaster.params.Points = { threshold: Math.max(0.01, propsRef.current.pointSize * 2) };

      const mode = propsRef.current.display;
      const targets: THREE.Object3D[] = [];
      if (mode === "points" || mode === "both") targets.push(points);
      if (mode === "mesh" || mode === "both") targets.push(mesh);
      if (!targets.length) return;

      const hits = raycaster.intersectObjects(targets, false);
      if (!hits.length) return;
      const hit = hits[0];
      const p = hit.point;       // world frame

      // Show the ping at the hit point.
      ping.position.copy(p);
      ping.visible = true;
      pingT0 = performance.now();

      // Convert world hit -> camera frame for the pinhole projection.
      cameraPose.updateMatrixWorld();
      const pCam = p.clone().applyMatrix4(
        new THREE.Matrix4().copy(cameraPose.matrixWorld).invert()
      );
      const z = Math.max(pCam.z, 1e-3);
      const u_inf = meta.fx_infer * (pCam.x / z) + meta.cx_infer;
      const v_inf = meta.fy_infer * (pCam.y / z) + meta.cy_infer;
      const x = Math.round(Math.max(0, Math.min(meta.infer_w - 1, u_inf)));
      const y = Math.round(Math.max(0, Math.min(meta.infer_h - 1, v_inf)));
      streamRef.current.samClick(x, y);
    };
    renderer.domElement.addEventListener("mousedown", onCanvasMouseDown);
    renderer.domElement.addEventListener("click", onCanvasClick);

    // ── Camera-pose gizmos ──────────────────────────────────────────────
    // Two TransformControls attached to cameraPose so translate arrows and
    // rotate rings are visible at the same time. Sized differently so the
    // rings sit outside the arrows and don't fight for clicks.
    // Two TransformControls (translate + rotate) attached to cameraPose.
    // Shown one at a time based on gizmoMode — they overlap badly when both
    // are visible (rings steal hits from arrows). Same UX as Viser when you
    // disable_sliders or disable_rotations.
    const gizmoT = new TransformControls(camera, renderer.domElement);
    gizmoT.setMode("translate");
    gizmoT.setSpace("local");
    gizmoT.size = 1.0;
    gizmoT.attach(cameraPose);
    const gizmoR = new TransformControls(camera, renderer.domElement);
    gizmoR.setMode("rotate");
    gizmoR.setSpace("local");
    gizmoR.size = 1.0;
    gizmoR.attach(cameraPose);
    const gizmoTHelper = gizmoT.getHelper();
    const gizmoRHelper = gizmoR.getHelper();
    scene.add(gizmoTHelper);
    scene.add(gizmoRHelper);

    let suppressEcho = false;   // ignore server echo while we're dragging
    const onDraggingChanged = (ev: { value: unknown }) => {
      const dragging = !!ev.value;
      controls.enabled = !dragging;
      if (dragging) {
        suppressEcho = true;
      } else {
        // Wait one server round-trip before re-accepting echoes so the
        // local gizmo position doesn't fight the echo.
        setTimeout(() => { suppressEcho = false; }, 250);
      }
    };
    gizmoT.addEventListener("dragging-changed", onDraggingChanged);
    gizmoR.addEventListener("dragging-changed", onDraggingChanged);

    // Throttled sender so dragging doesn't spam the WS at 60 Hz.
    let lastSent = 0;
    const SEND_PERIOD_MS = 33;        // ~30 Hz
    const sendCalibFromGizmo = () => {
      const now = performance.now();
      if (now - lastSent < SEND_PERIOD_MS) return;
      lastSent = now;
      const pos: [number, number, number] = [
        cameraPose.position.x, cameraPose.position.y, cameraPose.position.z,
      ];
      // Three.js Quaternion -> intrinsic XYZ Euler (degrees), matching the
      // server-side convention in config._euler_xyz_deg_to_matrix
      // (R = Rx @ Ry @ Rz).
      const e = new THREE.Euler().setFromQuaternion(cameraPose.quaternion, "XYZ");
      const eul: [number, number, number] = [
        THREE.MathUtils.radToDeg(e.x),
        THREE.MathUtils.radToDeg(e.y),
        THREE.MathUtils.radToDeg(e.z),
      ];
      streamRef.current.setCamExtrinsics(pos, eul);
    };
    gizmoT.addEventListener("objectChange", sendCalibFromGizmo);
    gizmoR.addEventListener("objectChange", sendCalibFromGizmo);

    // ── World-frame handle gizmo ────────────────────────────────────────
    // Drag the world-origin handle to nudge T_world_camera *inversely* —
    // i.e. moving the world +X by Δ is equivalent to moving the camera −Δ
    // in world. Useful when you want to align the robot mesh under the
    // point cloud rather than the other way around.
    //
    // Implementation: keep a separate "worldHandle" Object3D at world
    // identity. On drag-start we snapshot the camera pose. On every
    // objectChange we read the handle's pose, compute T_camera_new =
    // T_handle.inverse() · T_camera_start, and send. On drag-end we
    // snap the handle back to identity so the next grab starts fresh.
    const worldHandle = new THREE.Group();
    worldHandle.name = "world-handle";
    scene.add(worldHandle);
    const worldGizmoT = new TransformControls(camera, renderer.domElement);
    worldGizmoT.setMode("translate");
    worldGizmoT.setSpace("world");   // operate in world frame, naturally
    worldGizmoT.size = 1.0;
    worldGizmoT.attach(worldHandle);
    const worldGizmoR = new TransformControls(camera, renderer.domElement);
    worldGizmoR.setMode("rotate");
    worldGizmoR.setSpace("world");
    worldGizmoR.size = 1.0;
    worldGizmoR.attach(worldHandle);
    const worldGizmoTHelper = worldGizmoT.getHelper();
    const worldGizmoRHelper = worldGizmoR.getHelper();
    scene.add(worldGizmoTHelper);
    scene.add(worldGizmoRHelper);

    // Camera pose at the moment of drag-start (frozen reference).
    const camPosStart = new THREE.Vector3();
    const camQuatStart = new THREE.Quaternion();

    const onWorldDraggingChanged = (ev: { value: unknown }) => {
      const dragging = !!ev.value;
      controls.enabled = !dragging;
      if (dragging) {
        suppressEcho = true;
        // Snapshot camera pose; reset handle so deltas are absolute.
        camPosStart.copy(cameraPose.position);
        camQuatStart.copy(cameraPose.quaternion);
        worldHandle.position.set(0, 0, 0);
        worldHandle.quaternion.identity();
      } else {
        // Snap handle back to identity for next grab.
        worldHandle.position.set(0, 0, 0);
        worldHandle.quaternion.identity();
        setTimeout(() => { suppressEcho = false; }, 250);
      }
    };
    worldGizmoT.addEventListener("dragging-changed", onWorldDraggingChanged);
    worldGizmoR.addEventListener("dragging-changed", onWorldDraggingChanged);

    // Reusable scratch objects to avoid GC.
    const handleQuatInv = new THREE.Quaternion();
    const camPosNew = new THREE.Vector3();
    const camQuatNew = new THREE.Quaternion();

    const sendCalibFromWorldGizmo = () => {
      const now = performance.now();
      if (now - lastSent < SEND_PERIOD_MS) return;
      lastSent = now;
      // Compute T_camera_new = T_handle.inverse() · T_camera_start.
      // Position: subtract handle position from camera-start, then rotate
      // back into world by handle's inverse quaternion.
      handleQuatInv.copy(worldHandle.quaternion).invert();
      camPosNew.copy(camPosStart)
               .sub(worldHandle.position)
               .applyQuaternion(handleQuatInv);
      camQuatNew.copy(handleQuatInv).multiply(camQuatStart);
      // Update local cameraPose immediately so the user sees the result;
      // server echo is suppressed during drag so this won't fight.
      cameraPose.position.copy(camPosNew);
      cameraPose.quaternion.copy(camQuatNew);
      const e = new THREE.Euler().setFromQuaternion(camQuatNew, "XYZ");
      streamRef.current.setCamExtrinsics(
        [camPosNew.x, camPosNew.y, camPosNew.z],
        [
          THREE.MathUtils.radToDeg(e.x),
          THREE.MathUtils.radToDeg(e.y),
          THREE.MathUtils.radToDeg(e.z),
        ],
      );
    };
    worldGizmoT.addEventListener("objectChange", sendCalibFromWorldGizmo);
    worldGizmoR.addEventListener("objectChange", sendCalibFromWorldGizmo);

    // ── EE-target gizmo (ee_pose controller) ────────────────────────────
    // A draggable handle at the EE target. The handle's pose is the
    // setpoint sent to the OSC controller. Visibility/enable is gated on
    // (controller==ee_pose && status==running) and the gizmoMode toggle.
    const eeTarget = new THREE.Group();
    eeTarget.name = "ee-target";
    // Small triad so the gizmo origin is visible even when no controls
    // are showing (good for sanity-checks while dragging).
    const eeAxes = new THREE.AxesHelper(0.08);
    eeTarget.add(eeAxes);
    scene.add(eeTarget);

    const eeGizmoT = new TransformControls(camera, renderer.domElement);
    eeGizmoT.setMode("translate");
    eeGizmoT.setSpace("world");
    eeGizmoT.size = 0.85;
    eeGizmoT.attach(eeTarget);
    const eeGizmoR = new TransformControls(camera, renderer.domElement);
    eeGizmoR.setMode("rotate");
    eeGizmoR.setSpace("world");
    eeGizmoR.size = 0.85;
    eeGizmoR.attach(eeTarget);
    const eeGizmoTHelper = eeGizmoT.getHelper();
    const eeGizmoRHelper = eeGizmoR.getHelper();
    scene.add(eeGizmoTHelper);
    scene.add(eeGizmoRHelper);

    let suppressEeEcho = false;
    const onEeDragging = (ev: { value: unknown }) => {
      const dragging = !!ev.value;
      controls.enabled = !dragging;
      if (dragging) {
        suppressEeEcho = true;
      } else {
        // Send one final sync immediately, then re-allow echo after RTT.
        sendEeFromGizmo(true);
        setTimeout(() => { suppressEeEcho = false; }, 250);
      }
    };
    eeGizmoT.addEventListener("dragging-changed", onEeDragging);
    eeGizmoR.addEventListener("dragging-changed", onEeDragging);

    let lastEeSent = 0;
    const EE_SEND_PERIOD_MS = 33;     // ~30 Hz
    const sendEeFromGizmo = (force = false) => {
      const now = performance.now();
      if (!force && now - lastEeSent < EE_SEND_PERIOD_MS) return;
      lastEeSent = now;
      const p = eeTarget.position;
      const q = eeTarget.quaternion;   // three uses xyzw on .quaternion
      streamRef.current.setEeTarget(
        [p.x, p.y, p.z],
        [q.x, q.y, q.z, q.w],
      );
    };
    eeGizmoT.addEventListener("objectChange", () => sendEeFromGizmo());
    eeGizmoR.addEventListener("objectChange", () => sendEeFromGizmo());

    let lastEeKey = "";

    // Resize
    const onResize = () => {
      const w = container.clientWidth, h = container.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      uniforms.uPxPerMeter.value = h; // rough perspective-px-per-meter at 1m
    };
    window.addEventListener("resize", onResize);

    let lastPointsSeq = -1, lastMeshSeq = -1, lastMaskSeq = -1;
    let frustumBuilt = false;
    let lastCalibKey = "";

    const animate = () => {
      // Build the frustum geometry the first time intrinsics arrive.
      const m = streamRef.current.meta;
      if (!frustumBuilt && m) {
        frustumBuilt = true;
        const scale = 0.3;
        const corners: [number, number, number][] = [
          [(0       - m.cx) / m.fx * scale, (0       - m.cy) / m.fy * scale, scale],
          [(m.rgb_w - m.cx) / m.fx * scale, (0       - m.cy) / m.fy * scale, scale],
          [(m.rgb_w - m.cx) / m.fx * scale, (m.rgb_h - m.cy) / m.fy * scale, scale],
          [(0       - m.cx) / m.fx * scale, (m.rgb_h - m.cy) / m.fy * scale, scale],
        ];
        const segs: number[] = [];
        const apex: [number, number, number] = [0, 0, 0];
        for (const c of corners) segs.push(...apex, ...c);
        for (let i = 0; i < 4; i++) segs.push(...corners[i], ...corners[(i + 1) % 4]);
        segs.push(...corners[0], ...corners[2]);
        segs.push(...corners[1], ...corners[3]);
        frustumGeom.setAttribute("position", new THREE.Float32BufferAttribute(segs, 3));
      }

      // Apply T_world_camera to the cameraPose group whenever calibration
      // changes (live updates flow via KIND_CAM_CALIB into camCalibRef).
      // Skip while we're dragging the gizmo: our own writes already update
      // cameraPose, and overwriting from the server echo would fight the drag.
      const cc0 = streamRef.current.camCalibRef.current;
      if (cc0 && !suppressEcho) {
        const cc = cc0.extrinsics;
        const key = `${cc.pos_world.join(",")}|${cc.quat_wxyz.join(",")}`;
        if (key !== lastCalibKey) {
          lastCalibKey = key;
          cameraPose.position.set(cc.pos_world[0], cc.pos_world[1], cc.pos_world[2]);
          // wxyz -> xyzw
          cameraPose.quaternion.set(
            cc.quat_wxyz[1], cc.quat_wxyz[2], cc.quat_wxyz[3], cc.quat_wxyz[0]
          );
        }
      }

      // Toggles (read via ref so prop changes apply without remount)
      const p = propsRef.current;
      points.visible = p.display === "points" || p.display === "both";
      mesh.visible = p.display === "mesh" || p.display === "both";
      worldAxes.visible = p.showWorldAxes;   // RGB axes at world origin
      cameraPose.visible = p.showCamera;     // camera body (frustum + cam-local axes)
      // Gizmos follow cameraPose's visibility, with mode picking which one.
      const tOn = p.showCamera && p.gizmoMode === "translate";
      const rOn = p.showCamera && p.gizmoMode === "rotate";
      gizmoTHelper.visible = tOn;
      gizmoRHelper.visible = rOn;
      gizmoT.enabled = tOn;
      gizmoR.enabled = rOn;
      // World handle gizmo (independent visibility, same mode).
      const wtOn = p.showWorldHandle && p.gizmoMode === "translate";
      const wrOn = p.showWorldHandle && p.gizmoMode === "rotate";
      worldGizmoTHelper.visible = wtOn;
      worldGizmoRHelper.visible = wrOn;
      worldGizmoT.enabled = wtOn;
      worldGizmoR.enabled = wrOn;

      // EE-target gizmo: only shown for the active ee_pose controller.
      const cs = streamRef.current.controllerState;
      const eeOn = !!(cs && cs.current === "ee_pose" && cs.status === "running");
      eeTarget.visible = eeOn;
      const etOn = eeOn && p.gizmoMode === "translate";
      const erOn = eeOn && p.gizmoMode === "rotate";
      eeGizmoTHelper.visible = etOn;
      eeGizmoRHelper.visible = erOn;
      eeGizmoT.enabled = etOn;
      eeGizmoR.enabled = erOn;

      // Pull the latest target from the server when not dragging. The
      // server publishes shm_qtarget every 1 s plus on every controller
      // transition; the gizmo snaps to the seeded value on engage.
      if (eeOn && !suppressEeEcho && cs?.ee_target) {
        const t = cs.ee_target;
        const key = `${t.pos.join(",")}|${t.quat_xyzw.join(",")}`;
        if (key !== lastEeKey) {
          lastEeKey = key;
          eeTarget.position.set(t.pos[0], t.pos[1], t.pos[2]);
          eeTarget.quaternion.set(
            t.quat_xyzw[0], t.quat_xyzw[1], t.quat_xyzw[2], t.quat_xyzw[3],
          );
        }
      }
      if (!eeOn) lastEeKey = "";   // re-seed next time it engages
      uniforms.uBaseSize.value = p.pointSize;
      uniforms.uInversePerspective.value = p.inversePerspective ? 1.0 : 0.0;

      // Update points
      const pf = stream.pointsRef.current;
      if (pf && pf.seq !== lastPointsSeq) {
        lastPointsSeq = pf.seq;
        const colorsF = new Float32Array(pf.n * 3);
        for (let i = 0; i < pf.n * 3; i++) colorsF[i] = pf.rgb[i] / 255;
        const maskF = new Float32Array(pf.n);
        for (let i = 0; i < pf.n; i++) maskF[i] = pf.mask[i] ? 1.0 : 0.0;
        pointsGeom.setAttribute("position", new THREE.BufferAttribute(pf.xyz, 3));
        pointsGeom.setAttribute("color", new THREE.BufferAttribute(colorsF, 3));
        pointsGeom.setAttribute("mask", new THREE.BufferAttribute(maskF, 1));
        pointsGeom.computeBoundingSphere();
      }

      // Update bbox from mask frames (latest meta lives behind streamRef)
      const mr = streamRef.current.maskRef.current;
      if (mr && mr.seq !== lastMaskSeq) {
        lastMaskSeq = mr.seq;
        if (mr.hasBox) setBBox(mr.boxMin, mr.boxMax);
      }
      bboxLines.visible =
        propsRef.current.showBBox &&
        !!streamRef.current.maskRef.current?.hasBox;

      // Update mesh
      const mf = stream.meshRef.current;
      if (mf && mf.seq !== lastMeshSeq) {
        lastMeshSeq = mf.seq;
        const colorsF = new Float32Array(mf.nv * 3);
        for (let i = 0; i < mf.nv * 3; i++) colorsF[i] = mf.rgb[i] / 255;
        // Mesh-grid mask: identity index into the most recent SAM mask.
        const msrc = streamRef.current.maskRef.current;
        const maskF = new Float32Array(mf.nv);
        if (msrc && msrc.mask.length === mf.nv) {
          for (let i = 0; i < mf.nv; i++) maskF[i] = msrc.mask[i] ? 1.0 : 0.0;
        }
        meshGeom.setAttribute("position", new THREE.BufferAttribute(mf.xyz, 3));
        meshGeom.setAttribute("color", new THREE.BufferAttribute(colorsF, 3));
        meshGeom.setAttribute("mask", new THREE.BufferAttribute(maskF, 1));
        // Only attach normals when the backend ships them; otherwise remove
        // the attribute so a previous lit-mode mesh doesn't keep stale data.
        if (mf.normal && mf.normal.length === mf.nv * 3) {
          meshGeom.setAttribute("aNormal", new THREE.BufferAttribute(mf.normal as Float32Array, 3));
          meshMat.uniforms.uUseNormals.value = 1.0;
        } else {
          if (meshGeom.getAttribute("aNormal")) meshGeom.deleteAttribute("aNormal");
          meshMat.uniforms.uUseNormals.value = 0.0;
        }
        meshGeom.setIndex(new THREE.BufferAttribute(mf.faces, 1));
        meshGeom.computeBoundingSphere();
      }

      // Robot — apply geometry first (one-shot), then per-frame transforms.
      const rg = streamRef.current.robotGeomRef.current;
      if (rg && rg.seq !== lastRobotGeomSeq) {
        lastRobotGeomSeq = rg.seq;
        const eeIdx = streamRef.current.meta?.robot?.ee_body_idx ?? -1;
        robot.setGeometry(rg.bodies, rg.meshes, rg.geoms, rg.blob, eeIdx);
      }
      const rx = streamRef.current.robotXformRef.current;
      if (rx && rx.seq !== lastRobotXformSeq) {
        lastRobotXformSeq = rx.seq;
        robot.setTransforms(rx.xpos, rx.xquat, rx.nbody);
      }
      robot.setVisible(propsRef.current.showRobot);
      // Suppress the robot-tree EE axes (parented under the mesh, so they
      // inherit showRobot). We draw scene-level EE axes from ee_pose
      // instead — independent of the robot mesh visibility.
      robot.setEeAxesVisible(false);
      // Update the scene-level EE axes from the live ee_pose broadcast.
      const eePose = streamRef.current.controllerState?.ee_pose;
      if (eePose) {
        eeLiveAxes.position.set(eePose.pos[0], eePose.pos[1], eePose.pos[2]);
        eeLiveAxes.quaternion.set(
          eePose.quat_xyzw[0], eePose.quat_xyzw[1],
          eePose.quat_xyzw[2], eePose.quat_xyzw[3],
        );
        eeLiveAxes.visible = propsRef.current.showEeAxes;
      } else {
        eeLiveAxes.visible = false;
      }

      // Animate the click-feedback ping (grow + fade over 400ms).
      if (ping.visible) {
        const t = (performance.now() - pingT0) / 400;
        if (t >= 1) {
          ping.visible = false;
          pingMat.opacity = 0;
        } else {
          const s = 0.6 + 1.4 * t;
          ping.scale.set(s, s, s);
          pingMat.opacity = 0.8 * (1 - t);
        }
      }

      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    };
    let raf = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      renderer.domElement.removeEventListener("mousedown", onCanvasMouseDown);
      renderer.domElement.removeEventListener("click", onCanvasClick);
      // Detach + remove helpers. Skip TransformControls.dispose() — in
      // three r169 it calls this.traverse() but TransformControls extends
      // Controls (not Object3D), so it throws.
      for (const g of [gizmoT, gizmoR, worldGizmoT, worldGizmoR,
                       eeGizmoT, eeGizmoR]) g.detach();
      for (const h of [gizmoTHelper, gizmoRHelper, worldGizmoTHelper, worldGizmoRHelper,
                       eeGizmoTHelper, eeGizmoRHelper]) {
        scene.remove(h);
        if (typeof (h as { dispose?: () => void }).dispose === "function") {
          (h as { dispose: () => void }).dispose();
        }
      }
      scene.remove(worldHandle);
      scene.remove(eeTarget);
      eeAxes.dispose();
      controls.dispose();
      renderer.dispose();
      pointsMat.dispose();
      meshMat.dispose();
      pointsGeom.dispose();
      meshGeom.dispose();
      frustumGeom.dispose();
      frustumMat.dispose();
      bboxGeom.dispose();
      bboxMat.dispose();
      pingGeom.dispose();
      pingMat.dispose();
      robot.dispose();
      scene.remove(eeLiveAxes);
      eeLiveAxes.dispose();
      container.removeChild(renderer.domElement);
    };
    // We intentionally only run this effect once; values are read via refs/closure
    // and the latest props are picked up each animation frame through the closures
    // above (display, pointSize, depthSizeFactor, showCamera).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return <div ref={containerRef} style={{ width: "100%", height: "100%" }} />;
}
