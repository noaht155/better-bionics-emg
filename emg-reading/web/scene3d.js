// Shared three.js setup for the 3D pages (contact.js, vr.js): renderer, lights, camera in the sims' z-up frame, MuJoCo
// geoms as meshes and an orbit view turned with the right mouse button and zoomed with the wheel.
import * as THREE from "./vendor/three.module.min.js";

export { THREE };
export const color = (name) => new THREE.Color(cssVar(name));
export const rad = (d) => (d * Math.PI) / 180;

// shadowHalf: half the width (m) the sun's shadow covers round the origin
export function createView(canvas, shadowHalf = 0.5) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const world = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 20);
  camera.up.set(0, 0, 1);
  const bodies = new THREE.Group();
  world.add(bodies);
  world.background = color("--bg");
  world.add(new THREE.HemisphereLight(color("--text"), color("--bg"), 1.2));
  const sun = new THREE.DirectionalLight(color("--text"), 2.2);
  sun.position.set(0.3, -0.4, 1.0);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  const s = shadowHalf;
  Object.assign(sun.shadow.camera, { left: -s, right: s, top: s, bottom: -s, near: 0.1, far: 3 });
  world.add(sun);

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  return { renderer, world, camera, bodies, resize, render: () => renderer.render(world, camera) };
}

export function geometryFor(g) {
  const [a, b, c] = g.size;
  switch (g.type) {
    // MuJoCo's capsules and cylinders run along their z axis, three.js's along y
    case "capsule": return new THREE.CapsuleGeometry(a, 2 * b, 6, 16).rotateX(Math.PI / 2);
    case "cylinder": return new THREE.CylinderGeometry(a, a, 2 * b, 40).rotateX(Math.PI / 2);
    case "sphere": return new THREE.SphereGeometry(a, 32, 16);
    case "ellipsoid": return new THREE.SphereGeometry(1, 24, 12).scale(a, b, c);
    case "box": return new THREE.BoxGeometry(2 * a, 2 * b, 2 * c);
    case "plane": return new THREE.PlaneGeometry(2 * a, 2 * b);
    case "mesh": {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(g.vertices.flat(), 3));
      geo.setIndex(g.faces.flat());
      geo.computeVertexNormals();
      return geo;
    }
  }
  return null;
}

const rot = new THREE.Matrix4();
// Puts a mesh at a geom's position and 3 x 3 row-major orientation
export function placeMesh(mesh, pos, m) {
  mesh.position.fromArray(pos);
  rot.set(m[0], m[1], m[2], 0, m[3], m[4], m[5], 0, m[6], m[7], m[8], 0, 0, 0, 0, 1);
  mesh.quaternion.setFromRotationMatrix(rot);
}

// Orbit camera round view.target: yaw and pitch in degrees, dist in m
export function orbit(camera, view) {
  const p = rad(view.pitch), y = rad(view.yaw);
  camera.position.set(Math.cos(p) * Math.cos(y), Math.cos(p) * Math.sin(y), Math.sin(p))
    .multiplyScalar(view.dist).add(view.target);
  camera.lookAt(view.target);
}
