import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { StreamState } from "./useStream";

type Props = {
  stream: StreamState;
  pointSize: number;
  inversePerspective: boolean;
  display: "points" | "mesh" | "both";
  showCamera: boolean;
  showBBox: boolean;
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

export function Viewer({ stream, pointSize, inversePerspective, display, showCamera, showBBox }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const propsRef = useRef({ pointSize, inversePerspective, display, showCamera, showBBox });
  propsRef.current = { pointSize, inversePerspective, display, showCamera, showBBox };
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
    camera.position.set(0, 0, -1.5);
    camera.up.set(0, -1, 0);
    camera.lookAt(0, 0, 0);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.update();

    // World axes at origin
    const axes = new THREE.AxesHelper(0.15);
    scene.add(axes);

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
    // Camera frustum (placeholder; sized later when meta arrives)
    const frustumGeom = new THREE.BufferGeometry();
    const frustumMat = new THREE.LineBasicMaterial({ color: 0x33cc55 });
    const frustum = new THREE.LineSegments(frustumGeom, frustumMat);
    scene.add(frustum);

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

    // Mesh — custom ShaderMaterial so we can tint masked vertices the same
    // way the points are tinted.
    const meshGeom = new THREE.BufferGeometry();
    meshGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
    meshGeom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(0), 3));
    meshGeom.setAttribute("mask", new THREE.BufferAttribute(new Float32Array(0), 1));
    meshGeom.setIndex(new THREE.BufferAttribute(new Uint32Array(0), 1));
    const meshMat = new THREE.ShaderMaterial({
      uniforms: {
        uHighlight: uniforms.uHighlight,
        uHighlightAmt: uniforms.uHighlightAmt,
      },
      vertexShader: /* glsl */ `
        attribute vec3 color;
        attribute float mask;
        varying vec3 vColor;
        varying float vMask;
        void main() {
          vColor = color;
          vMask = mask;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        varying vec3 vColor;
        varying float vMask;
        uniform vec3 uHighlight;
        uniform float uHighlightAmt;
        void main() {
          vec3 c = mix(vColor, uHighlight, vMask * uHighlightAmt);
          gl_FragColor = vec4(c, 1.0);
        }`,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(meshGeom, meshMat);
    scene.add(mesh);

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
    let metaApplied = false;

    const animate = () => {
      // Apply meta -> frustum once
      if (!metaApplied && streamRef.current.meta) {
        metaApplied = true;
        const m = streamRef.current.meta;
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

      // Toggles (read via ref so prop changes apply without remount)
      const p = propsRef.current;
      points.visible = p.display === "points" || p.display === "both";
      mesh.visible = p.display === "mesh" || p.display === "both";
      axes.visible = p.showCamera;
      frustum.visible = p.showCamera;
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
        meshGeom.setIndex(new THREE.BufferAttribute(mf.faces, 1));
        meshGeom.computeBoundingSphere();
      }

      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    };
    let raf = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
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
      container.removeChild(renderer.domElement);
    };
    // We intentionally only run this effect once; values are read via refs/closure
    // and the latest props are picked up each animation frame through the closures
    // above (display, pointSize, depthSizeFactor, showCamera).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return <div ref={containerRef} style={{ width: "100%", height: "100%" }} />;
}
