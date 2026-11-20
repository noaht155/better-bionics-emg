// 3D contact page on the Control tab: draws sim.py's world with three.js, moves the hand from the keys and mouse
// and slides the object over the table. The physics runs on the server (tools.Contact), this page only draws it.
import { THREE, color, createView, geometryFor, orbit, placeMesh, rad } from "./scene3d.js";

const FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"];
// A finger pressing this hard shows fully in the effort colour
const TINT_FULL_N = 10;
const MOVE_M_S = 0.15;
const SLOW = 0.25;
const TURN_DEG_PER_PX = 0.4;
const ROLL_DEG_S = 90;
const VIEW_DEG_PER_PX = 0.3;
// Hand pose and object drags sent at most this often
const SEND_MS = 33;
const MOVE_KEYS = ["w", "a", "s", "d", "q", "e", "z", "x"];
export const CONTACT_PARAMS = [
  { key: "mass_scale", label: "Mass x", min: 0.25, max: 3, step: 0.25, help: "the object's mass as a multiple of its own: can 350 g, ball 58 g, block 120 g, small cube 10 g, pen 12 g" },
  { key: "max_force", label: "Hand strength N", min: 2, max: 40, step: 1, help: "force each finger presses with at full effort" },
  { key: "assist", label: "Grasp assist", help: "holds the object to the palm while the thumb and a finger press on it hard enough for friction to carry its weight, for grasps physics alone can't hold. A grip too light for the weight still drops it" },
];

const cx = { open: false, ws: null, meshes: [], version: -1, hand: null, keys: new Set(), drag: null,
  params: null, lastSend: 0, handMoved: false, lastFrame: 0 };
const view = { yaw: -55, pitch: 28, dist: 0.65, target: new THREE.Vector3(0, -0.09, 0.05) };

const canvas = $("ctl-contact-canvas");
const { renderer, world, camera, bodies, resize } = createView(canvas);
const HAND = color("--hand-solid");
const PRESSED = color("--effort");
const grid = new THREE.GridHelper(0.9, 18, color("--line-strong"), color("--line"));
grid.rotation.x = Math.PI / 2;
grid.position.z = 0.0005;
world.add(grid);

const placeCamera = () => orbit(camera, view);

function buildScene(scene) {
  bodies.clear();
  cx.meshes = scene.geoms.map((g) => {
    const geo = geometryFor(g);
    if (!geo) return null;
    const base = color(g.role === "table" ? "--raised" : g.role === "object" ? "--sim-object" : "--hand-solid");
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: base, roughness: 0.65 }));
    mesh.castShadow = g.role !== "table";
    mesh.receiveShadow = true;
    const part = g.name?.split("_")[0];
    mesh.userData = { role: g.role, finger: FINGER_NAMES.indexOf(part), palm: part === "palm" };
    bodies.add(mesh);
    return mesh;
  });
  cx.version = scene.version;
  cx.params = { ...scene.params };
  $("ctl-contact-object").innerHTML = scene.objects.map((o) => `<option>${o}</option>`).join("");
  $("ctl-contact-object").value = scene.object;
  showParams(CONTACT_PARAMS, "ctl-contact-", cx.params);
}

async function loadScene() {
  const res = await fetch("/api/contact/scene");
  if (res.ok) buildScene(await res.json());
}

function applyFrame(f) {
  cx.lastFrame = performance.now();
  if (!f.running) {
    $("ctl-contact-status").textContent = "The simulation stopped.";
    return;
  }
  if (f.version !== cx.version) {
    cx.version = f.version;
    loadScene();
    return;
  }
  // The Quest app changes the same settings
  if (f.params && JSON.stringify(f.params) !== JSON.stringify(cx.params)) {
    cx.params = { ...f.params };
    showParams(CONTACT_PARAMS, "ctl-contact-", cx.params);
  }
  const r = f.readout;
  cx.meshes.forEach((mesh, i) => {
    if (!mesh) return;
    placeMesh(mesh, f.pos[i], f.mat[i]);
    const { finger, palm, role } = mesh.userData;
    if (role !== "hand") return;
    const force = finger >= 0 ? r.finger_force[finger] : palm ? r.palm_force : 0;
    mesh.material.color.lerpColors(HAND, PRESSED, Math.min(force / TINT_FULL_N, 1));
  });
  if (!cx.hand) {
    const [p, q] = f.hand;
    cx.hand = { pos: new THREE.Vector3(...p), quat: new THREE.Quaternion(q[1], q[2], q[3], q[0]) };
  }
  showReadout(r, f.control, f.quest);
}

function showReadout(r, control, quest) {
  const needed = r.weight / r.friction;
  const fingers = FINGER_NAMES.map((n, i) => `${n} ${r.finger_force[i].toFixed(1)}`).join(", ");
  $("ctl-contact-readout").innerHTML = `
    <div class="contact-force"><b>${r.grip_force.toFixed(1)} N</b><span>grip force</span></div>
    <div class="contact-force"><b>${needed.toFixed(1)} N</b><span>to hold its weight</span></div>
    <p class="muted">Weight ${r.weight.toFixed(2)} N, friction ${r.friction}: the grip holds while grip force x
      friction is at least the weight. Per finger (N): ${fingers}, palm ${r.palm_force.toFixed(1)}.
      ${r.assisted ? "The assist holds the object." : ""}</p>`;
  $("ctl-contact-status").textContent = control
    ? `${control.grip ?? "relaxed"}${control.locked ? ", locked" : ""}, effort ${control.effort?.toFixed(2)}`
    : "No model runs: start one, or Simulate, on Grip control to move the fingers.";
  if (quest) {
    $("ctl-contact-status").textContent +=
      ` Quest on USB: its grip force (${quest.grip_force.toFixed(1)} N) drives the motors.`;
  }
}

// Hand and object placement

function viewAxes() {
  const y = rad(view.yaw);
  // Away from the viewer along the table, and to the viewer's right
  return { ahead: new THREE.Vector3(-Math.cos(y), -Math.sin(y), 0), right: new THREE.Vector3(-Math.sin(y), Math.cos(y), 0) };
}

const Z = new THREE.Vector3(0, 0, 1);
function moveHand(dt) {
  if (!cx.hand || !cx.keys.size) return;
  const k = cx.keys;
  const step = MOVE_M_S * dt * (k.has("shift") ? SLOW : 1);
  const { ahead, right } = viewAxes();
  const move = new THREE.Vector3();
  if (k.has("w")) move.add(ahead);
  if (k.has("s")) move.sub(ahead);
  if (k.has("d")) move.add(right);
  if (k.has("a")) move.sub(right);
  if (k.has("e")) move.add(Z);
  if (k.has("q")) move.sub(Z);
  cx.hand.pos.addScaledVector(move, step);
  const roll = (k.has("x") ? 1 : 0) - (k.has("z") ? 1 : 0);
  if (roll) {
    // Around the forearm, the hand's own wrist to fingers axis
    const turn = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), rad(ROLL_DEG_S * dt * roll * (k.has("shift") ? SLOW : 1)));
    cx.hand.quat.multiply(turn);
  }
  cx.handMoved = true;
}

function turnHand(dx, dy) {
  const { right } = viewAxes();
  const yaw = new THREE.Quaternion().setFromAxisAngle(Z, rad(dx * TURN_DEG_PER_PX));
  const pitch = new THREE.Quaternion().setFromAxisAngle(right, rad(dy * TURN_DEG_PER_PX));
  cx.hand.quat.premultiply(pitch).premultiply(yaw).normalize();
  cx.handMoved = true;
}

function sendHand(now) {
  if (!cx.handMoved || now - cx.lastSend < SEND_MS || cx.ws?.readyState !== WebSocket.OPEN) return;
  const q = cx.hand.quat;
  cx.ws.send(JSON.stringify({ hand: { pos: cx.hand.pos.toArray(), quat: [q.w, q.x, q.y, q.z] } }));
  cx.lastSend = now;
  cx.handMoved = false;
}

const raycaster = new THREE.Raycaster();
function pointer(e) {
  const box = canvas.getBoundingClientRect();
  return new THREE.Vector2(((e.clientX - box.left) / box.width) * 2 - 1, -((e.clientY - box.top) / box.height) * 2 + 1);
}

canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("pointerdown", (e) => {
  canvas.focus();
  canvas.setPointerCapture(e.pointerId);
  const start = { x: e.clientX, y: e.clientY };
  if (e.button === 2) {
    cx.drag = { kind: "view", ...start };
    return;
  }
  raycaster.setFromCamera(pointer(e), camera);
  const hit = raycaster.intersectObjects(cx.meshes.filter((m) => m?.userData.role === "object"))[0];
  if (hit) {
    // Slides the object at the height it was picked up at
    const height = hit.object.position.z;
    cx.drag = { kind: "object", plane: new THREE.Plane(Z, -height), offset: hit.object.position.clone().sub(hit.point).setZ(0), last: 0 };
  } else if (cx.hand) {
    cx.drag = { kind: "hand", ...start };
  }
});
canvas.addEventListener("pointermove", (e) => {
  const d = cx.drag;
  if (!d) return;
  if (d.kind === "object") {
    const now = performance.now();
    if (now - d.last < SEND_MS) return;
    raycaster.setFromCamera(pointer(e), camera);
    const at = raycaster.ray.intersectPlane(d.plane, new THREE.Vector3());
    if (!at) return;
    d.last = now;
    post("/api/contact/place", { pos: at.add(d.offset).toArray() }).catch(() => {});
    return;
  }
  const dx = e.clientX - d.x, dy = e.clientY - d.y;
  d.x = e.clientX;
  d.y = e.clientY;
  if (d.kind === "view") {
    view.yaw -= dx * VIEW_DEG_PER_PX;
    view.pitch = Math.min(85, Math.max(5, view.pitch + dy * VIEW_DEG_PER_PX));
    placeCamera();
  } else {
    turnHand(dx, dy);
  }
});
const endDrag = () => { cx.drag = null; };
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  view.dist = Math.min(1.5, Math.max(0.2, view.dist * Math.exp(e.deltaY * 0.001)));
  placeCamera();
}, { passive: false });

document.addEventListener("keydown", (e) => {
  if (!cx.open || e.target.closest("input, select, textarea")) return;
  const key = e.key.toLowerCase();
  if (key === "shift") cx.keys.add("shift");
  if (MOVE_KEYS.includes(key) && !e.ctrlKey && !e.metaKey && !e.altKey) {
    e.preventDefault();
    cx.keys.add(key);
  }
});
document.addEventListener("keyup", (e) => { cx.keys.delete(e.key.toLowerCase()); });
// Keys held while the window loses focus never get their keyup
window.addEventListener("blur", () => cx.keys.clear());

let lastTick = performance.now();
function tick(now) {
  if (!cx.open) return;
  const dt = Math.min((now - lastTick) / 1000, 0.1);
  lastTick = now;
  moveHand(dt);
  sendHand(now);
  if (now - cx.lastFrame > 2000) $("ctl-contact-status").textContent = "Waiting for the simulation";
  renderer.render(world, camera);
  requestAnimationFrame(tick);
}

// Opening and closing the page

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws/contact`);
  ws.onmessage = (e) => applyFrame(JSON.parse(e.data));
  ws.onclose = () => { if (cx.open && cx.ws === ws) setTimeout(connect, 1000); };
  cx.ws = ws;
}

async function showContactPage(on) {
  $("ctl-main").hidden = on;
  $("ctl-contact-page").hidden = !on;
  cx.open = on;
  cx.keys.clear();
  if (!on) {
    const ws = cx.ws;
    cx.ws = null;
    ws?.close();
    post("/api/contact/stop").catch(() => {});
    return;
  }
  const scene = await postShowing("/api/contact/start", null, "ctl-contact-error");
  if (!scene || !cx.open) return;
  cx.hand = null;
  buildScene(scene);
  resize();
  placeCamera();
  connect();
  lastTick = performance.now();
  requestAnimationFrame(tick);
  canvas.focus();
}

$("ctl-contact-open").addEventListener("click", () => showContactPage(true));
$("ctl-contact-back").addEventListener("click", () => showContactPage(false));
$("ctl-contact-reset").addEventListener("click", () => postShowing("/api/contact/reset", null, "ctl-contact-error"));
$("ctl-contact-object").addEventListener("change", (e) => {
  postShowing("/api/contact/params", { object: e.target.value }, "ctl-contact-error");
});
buildParams($("ctl-contact-params"), CONTACT_PARAMS, "ctl-contact-", () => cx.params, () => {
  const { mass_scale, max_force, assist } = cx.params;
  postShowing("/api/contact/params", { mass_scale, max_force, assist }, "ctl-contact-error");
});
window.addEventListener("resize", () => { if (cx.open) resize(); });
