/**
 * Three.js renderer for an MJCF robot.
 *
 * Geometry comes once via KIND_ROBOT_GEOMETRY (per-body local meshes/primitives
 * + a kinematic tree). Per-frame transforms come via KIND_ROBOT_TRANSFORMS
 * (one world pos+quat per body). We build a flat group of body Object3Ds at
 * construction time and just update their position/quaternion each frame; no
 * geometry is re-allocated.
 *
 * MuJoCo's world is Z-up. The existing scene is in OpenCV camera frame
 * (Y-down, Z-forward). We expose the whole robot under a single root group
 * so a future calibration step can place it correctly relative to the camera.
 */

import * as THREE from "three";
import {
  GEOM_BOX, GEOM_CAPSULE, GEOM_CYLINDER, GEOM_ELLIPSOID, GEOM_MESH,
  GEOM_PLANE, GEOM_SPHERE,
  RobotBody, RobotGeom, RobotMeshIndex,
} from "./protocol";

export class RobotRenderer {
  readonly root: THREE.Group;
  private bodyNodes: THREE.Object3D[] = [];
  private materials: THREE.Material[] = [];
  private geometries: THREE.BufferGeometry[] = [];

  constructor() {
    this.root = new THREE.Group();
    this.root.name = "robot-root";
    // Z-up MuJoCo → leave identity; calibration will rotate later.
    this.root.visible = true;
  }

  /**
   * Build (or rebuild) the scene-graph from a one-shot geometry payload.
   * Disposes any previously-built nodes.
   */
  setGeometry(bodies: RobotBody[], meshes: RobotMeshIndex[],
              geoms: RobotGeom[], blob: ArrayBuffer): void {
    this.dispose();

    // Build per-body groups, indexed by body id. We use a flat layout (each
    // body parented directly to root) and set absolute world transforms each
    // frame, matching MuJoCo's xpos/xquat. This avoids depending on the
    // kinematic tree being topologically sorted.
    this.bodyNodes = [];
    for (let i = 0; i < bodies.length; i++) {
      const node = new THREE.Group();
      node.name = `body-${i}-${bodies[i].name}`;
      this.root.add(node);
      this.bodyNodes.push(node);
    }

    // Decode each unique mesh once into a BufferGeometry.
    const meshGeoms: THREE.BufferGeometry[] = meshes.map((m) => {
      const verts = new Float32Array(blob, m.vert_offset, m.vert_count * 3);
      const faces = new Uint32Array(blob, m.face_offset, m.face_count * 3);
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(verts, 3));
      g.setIndex(new THREE.BufferAttribute(faces, 1));
      g.computeVertexNormals();
      this.geometries.push(g);
      return g;
    });

    // Build a Mesh per geom, parented under its body node and offset by the
    // geom's local pos/quat (MuJoCo geoms are specified in body-local coords).
    for (const g of geoms) {
      const mat = new THREE.MeshLambertMaterial({
        color: new THREE.Color(g.color[0], g.color[1], g.color[2]),
        opacity: g.color[3],
        transparent: g.color[3] < 1.0,
        side: THREE.DoubleSide,
      });
      this.materials.push(mat);

      const obj = makeGeomMesh(g, mat, meshGeoms);
      if (!obj) continue;
      obj.position.set(g.pos[0], g.pos[1], g.pos[2]);
      // MuJoCo quat is wxyz; THREE expects xyzw.
      obj.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
      this.bodyNodes[g.body].add(obj);
    }
  }

  /**
   * Apply per-body world transforms (one pos+quat per body, in MuJoCo world
   * frame). Length checks against the previously-set geometry; mismatches are
   * silently ignored (a stale transform from a prior model will arrive once
   * before the new geometry catches up).
   */
  setTransforms(xpos: Float32Array, xquat: Float32Array, nbody: number): void {
    if (nbody !== this.bodyNodes.length) return;
    for (let i = 0; i < nbody; i++) {
      const n = this.bodyNodes[i];
      n.position.set(xpos[3 * i], xpos[3 * i + 1], xpos[3 * i + 2]);
      // wxyz -> xyzw
      n.quaternion.set(
        xquat[4 * i + 1], xquat[4 * i + 2], xquat[4 * i + 3], xquat[4 * i]
      );
    }
  }

  setVisible(v: boolean): void {
    this.root.visible = v;
  }

  dispose(): void {
    for (const n of this.bodyNodes) {
      while (n.children.length) {
        const c = n.children[0];
        n.remove(c);
      }
      this.root.remove(n);
    }
    this.bodyNodes = [];
    for (const g of this.geometries) g.dispose();
    this.geometries = [];
    for (const m of this.materials) m.dispose();
    this.materials = [];
  }
}


function makeGeomMesh(g: RobotGeom, mat: THREE.Material,
                      meshGeoms: THREE.BufferGeometry[]): THREE.Object3D | null {
  switch (g.type) {
    case GEOM_PLANE:
      // Planes in MuJoCo describe the floor; we already render our own. Skip.
      return null;
    case GEOM_SPHERE: {
      const geom = new THREE.SphereGeometry(g.size[0], 16, 12);
      return new THREE.Mesh(geom, mat);
    }
    case GEOM_CAPSULE: {
      // size[0]=radius, size[1]=half-length along z (MuJoCo).
      const geom = new THREE.CapsuleGeometry(g.size[0], 2 * g.size[1], 8, 16);
      const m = new THREE.Mesh(geom, mat);
      // CapsuleGeometry's axis is +Y; MuJoCo's is +Z. Rotate.
      m.rotation.x = Math.PI / 2;
      return m;
    }
    case GEOM_CYLINDER: {
      // size[0]=radius, size[1]=half-height along z.
      const geom = new THREE.CylinderGeometry(g.size[0], g.size[0], 2 * g.size[1], 24);
      const m = new THREE.Mesh(geom, mat);
      m.rotation.x = Math.PI / 2;
      return m;
    }
    case GEOM_ELLIPSOID: {
      const geom = new THREE.SphereGeometry(1, 16, 12);
      const m = new THREE.Mesh(geom, mat);
      m.scale.set(g.size[0], g.size[1], g.size[2]);
      return m;
    }
    case GEOM_BOX: {
      // MuJoCo size = half-extents.
      const geom = new THREE.BoxGeometry(2 * g.size[0], 2 * g.size[1], 2 * g.size[2]);
      return new THREE.Mesh(geom, mat);
    }
    case GEOM_MESH: {
      if (g.mesh == null) return null;
      const geom = meshGeoms[g.mesh];
      if (!geom) return null;
      return new THREE.Mesh(geom, mat);
    }
    default:
      return null;
  }
}
