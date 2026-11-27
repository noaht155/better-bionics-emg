// VR tab: what the person in the headset sees, drawn from the Quest app's own simulation (/ws/vr: its scene once,
// then poses and head at 30 Hz), with its settings and objects beside it. From their eyes, or an orbit view.
import { THREE, color, createView, geometryFor, orbit, placeMesh } from "./scene3d.js";
import { CONTACT_PARAMS, frictionColor } from "./contact.js";
import { createCopies, createOverlay } from "./vrpanels.js";

// A finger pressing this hard shows fully in the effort colour, as on the contact page
const TINT_FULL_N = 10;
const EYES_FOV = 75, ORBIT_FOV = 40;
const VIEW_DEG_PER_PX = 0.3;
const OBJECT_NAMES = { can: "Can", ball: "Ball", block: "Block", "small cube": "Cube", pen: "Pen" };

const canvas = $("vr-canvas");
const { world, camera, bodies, resize, render } = createView(canvas, 0.8);
const HAND = color("--hand-solid"), FREE = color("--sim-free-hand");
// Objects and the rail in their friction's colour as on the contact page; a headset built before friction was sent
// falls back to its surface, one built before surfaces to grey
const surfaceColor = (g) => (typeof g.friction === "number" ? frictionColor(g.friction)
  : color(g.surface ? `--surface-${g.surface}` : "--sim-object"));
const PRESSED = color("--effort"), HELD = color("--accent-orange");
const showOverlay = createOverlay(world);
const showCopies = createCopies($("vr-panels"), "vr-error", postShowing);
const vr = { open: false, ws: null, meshes: [], params: {}, lastView: 0, mode: "eyes", turning: null };
// Headset video (/ws/vr-video): the Quest's screen is both eyes side by side, the left one is drawn
const videoCanvas = $("vr-video");
const video = { ws: null, decoder: null, codec: null, frames: 0, at: 0, error: "" };
// Behind the user's seat, looking over the desk
const orbitView = { yaw: 90, pitch: 32, dist: 1.1, target: new THREE.Vector3(0, -0.1, 0.05) };

function material(c, rough = 0.65) {
  return new THREE.MeshStandardMaterial({ color: c, roughness: rough });
}

function buildScene(scene) {
  bodies.clear();
  vr.meshes = scene.geoms.map((g) => {
    let geo;
    let base;
    if (g.type === "plane") {
      // The floor plane is infinite in the sim; drawn as 8 m with the page's grid
      geo = new THREE.PlaneGeometry(8, 8);
      base = color("--panel");
    } else {
      geo = geometryFor(g);
      base = g.role === "object" || (g.role === "rail" && g.type === "cylinder") ? surfaceColor(g)
        : g.role === "static" || g.role === "rail" ? color("--raised") : g.hand === 0 ? HAND : FREE;
    }
    if (!geo) return null;
    const mesh = new THREE.Mesh(geo, material(base));
    if (g.type === "plane") {
      const grid = new THREE.GridHelper(8, 16, color("--line"), color("--line"));
      grid.rotation.x = Math.PI / 2;
      grid.position.z = 0.001;
      mesh.add(grid);
    }
    mesh.castShadow = g.role !== "static";
    mesh.receiveShadow = true;
    mesh.userData = { role: g.role, hand: g.hand, finger: g.finger, obj: g.obj, base };
    bodies.add(mesh);
    return mesh;
  });
}

function applyView(v, readout) {
  vr.lastView = performance.now();
  vr.meshes.forEach((mesh, i) => {
    if (!mesh) return;
    placeMesh(mesh, v.pos.slice(3 * i, 3 * i + 3), v.mat.slice(9 * i, 9 * i + 9));
    const { role, hand, finger, obj, base } = mesh.userData;
    if (role === "object") {
      mesh.visible = !v.parked[obj];
      mesh.material.color.copy(v.held.includes(obj) ? base.clone().lerp(HELD, 0.35) : base);
    } else if (role === "hand" && hand === 0 && readout) {
      const force = finger >= 0 ? readout.finger_force[finger] : readout.palm_force;
      mesh.material.color.lerpColors(HAND, PRESSED, Math.min(force / TINT_FULL_N, 1));
    }
  });
  if (vr.mode === "eyes") {
    const h = v.head;
    camera.position.set(h[0], h[1], h[2]);
    camera.up.set(h[6], h[7], h[8]);
    camera.lookAt(h[0] + h[3], h[1] + h[4], h[2] + h[5]);
  }
}

function showReadout(readout) {
  if (!readout) {
    $("vr-readout").innerHTML = "";
    return;
  }
  $("vr-readout").innerHTML = `
    <div class="contact-force"><b>${readout.grip_force.toFixed(1)} N</b><span>prosthetic grip</span></div>
    <div class="contact-force"><b>${(readout.free_grip_force ?? 0).toFixed(1)} N</b><span>free hand grip</span></div>`;
}

// The headset's tracking (Tracking.cs): each controller and real hand, what the free hand goes by, and a list of the
// controllers' changes with the time they happened, newest first
function trackingSide(name, s) {
  const lost = s.lost_s > 0 ? ` for ${s.lost_s.toFixed(1)} s` : "";
  const last = s.losses ? ` (last ${s.last_loss_s.toFixed(1)} s)` : "";
  const hand = s.hand ? `seen${s.wrist_cm >= 0 ? `, ${Math.round(s.wrist_cm)} cm from it` : ""}` : "not seen";
  return `${name} controller ${s.controller}${lost}, lost ${s.losses}x${last}, hand ${hand}.`;
}

function showTracking(view) {
  const t = view?.tracking;
  if (!t) {
    $("vr-tracking").textContent = view ? "The headset runs a build from before tracking was sent." : "";
    return;
  }
  $("vr-tracking").textContent = [trackingSide("Left", t.left), trackingSide("Right", t.right),
    `Free hand goes by: ${t.free}.`, t.adjust && `${t.adjust}.`, `Hands and controllers together: ${t.multimodal}.`].filter(Boolean).join(" ");
  const log = vr.trackingLog || (vr.trackingLog = []);
  for (const [name, s] of [["Left", t.left], ["Right", t.right]]) {
    const before = vr.trackingWas?.[name];
    if (before !== undefined && before !== s.controller) {
      log.unshift(`${new Date().toLocaleTimeString()}  ${name} controller ${before} to ${s.controller}`);
    }
  }
  vr.trackingWas = { Left: t.left.controller, Right: t.right.controller };
  log.length = Math.min(log.length, 12);
  const list = $("vr-tracking-log");
  if (list.childElementCount !== log.length || list.firstChild?.textContent !== log[0]) {
    list.replaceChildren(...log.map((line) => Object.assign(document.createElement("li"), { textContent: line })));
  }
}

function batteryText(h) {
  return h ? ` Battery ${h.level} %${h.charging ? " charging" : ""}, ${Math.round(h.temp_c)} °C.` : "";
}

// Open the headset's menu on a page, close it, or start the tour, from the buttons the headset's pages make
function showMenuPages(pages) {
  const key = pages ? pages.join("|") : "";
  if (key === vr.menuPages) return;
  vr.menuPages = key;
  const box = $("vr-menu-pages");
  box.innerHTML = "";
  if (!pages) return;
  const add = (text, fn) => {
    const b = document.createElement("button");
    b.textContent = text;
    b.addEventListener("click", fn);
    box.append(b);
  };
  for (const page of pages) add(`Open ${page}`, () => postShowing("/api/vr/menu", { page }, "vr-error"));
  add("Close the menu", () => postShowing("/api/vr/menu", { page: "" }, "vr-error"));
  add("Start the tour", () => postShowing("/api/vr/tour", null, "vr-error"));
}

function applyFrame(msg) {
  if (msg.scene) {
    buildScene(msg.scene);
    return;
  }
  if (msg.params && JSON.stringify(msg.params) !== JSON.stringify(vr.params)) {
    vr.params = { ...msg.params };
    showParams(CONTACT_PARAMS, "vr-", vr.params);
  }
  if (msg.view && vr.meshes.length) applyView(msg.view, msg.readout);
  showOverlay(msg.view?.panels, msg.view?.ray);
  const shown = showCopies(msg.view?.panels);
  $("vr-panels-note").textContent = !msg.view ? "" : !msg.view.panels ? "The headset runs a build from before the menu was sent."
    : shown ? "Click a control to make it pulse in the headset." : "";
  showReadout(msg.readout);
  vr.battery = batteryText(msg.headset) + (msg.rtt_ms != null ? ` Link ${Math.round(msg.rtt_ms)} ms round trip.` : "");
  showMenuPages(msg.view?.menu_pages);
  showTracking(msg.view);
  if (msg.keep_video !== undefined && document.activeElement !== $("vr-keep-video")) $("vr-keep-video").checked = msg.keep_video;
  if (vr.mode !== "video") $("vr-status").textContent = !msg.view
    ? "Waiting for the headset: plug the Quest in over USB and open BB Contact."
    : `Headset sim ${msg.readout ? msg.readout.speed.toFixed(1) : "?"}x real time${msg.view.menu ? ", menu open" : ""}.`
      + vr.battery;
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws/vr`);
  vr.ws = ws;
  ws.onmessage = (e) => { if (vr.ws === ws) applyFrame(JSON.parse(e.data)); };
  ws.onclose = () => { if (vr.ws === ws && vr.open) setTimeout(connect, 1000); };
}

function closeVideo() {
  if (video.clip) video.clip.stop();
  const { ws, decoder } = video;
  video.ws = video.decoder = video.codec = null;
  ws?.close();
  if (decoder && decoder.state !== "closed") decoder.close();
}

// The headset records what goes to its lenses: each eye pre-warped (barrel) so the lens undoes it. The left eye is
// flattened back on the GPU: an output point u (in lens radii from the lens centre) comes from u (1 + K r^2) in the
// recording. K, the lens centre and the radius were fitted by eye on a 1280x720 recording until the room's straight
// edges came out straight (2026-11-27: -0.10 to -0.14 straight, -0.18 bends the other way). SPAN is how far out the
// view goes, in lens radii, kept inside the round eye image
const LENS = { K: -0.12, X: 0.254, Y: 0.5, R: 0.5, SPAN: 0.8 };
const FLAT_SHADER = `precision mediump float;
  varying vec2 p; uniform sampler2D tex; uniform vec2 size, centre, extent; uniform float radius, k;
  void main() {
    vec2 u = p * extent;
    vec2 s = centre + vec2(u.x, -u.y) * (1.0 + k * dot(u, u)) * radius;
    gl_FragColor = texture2D(tex, s / size);
  }`;
let flat = null;

function flatView(gl) {
  const shader = (type, text) => {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, text);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(sh));
    return sh;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, shader(gl.VERTEX_SHADER, "attribute vec2 a; varying vec2 p; void main() { p = a; gl_Position = vec4(a, 0, 1); }"));
  gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, FLAT_SHADER));
  gl.linkProgram(prog);
  gl.useProgram(prog);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const a = gl.getAttribLocation(prog, "a");
  gl.enableVertexAttribArray(a);
  gl.vertexAttribPointer(a, 2, gl.FLOAT, false, 0, 0);
  gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
  for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
    [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
  const at = (name) => gl.getUniformLocation(prog, name);
  return { gl, size: at("size"), centre: at("centre"), extent: at("extent"), radius: at("radius"), k: at("k") };
}

function drawVideoFrame(frame) {
  const w = videoCanvas.clientWidth, h = videoCanvas.clientHeight;
  if (videoCanvas.width !== w || videoCanvas.height !== h) Object.assign(videoCanvas, { width: w, height: h });
  // Kept so a picture or a clip can be saved from the canvas
  flat ??= flatView(videoCanvas.getContext("webgl", { preserveDrawingBuffer: true }));
  const { gl } = flat;
  const fw = frame.displayWidth, fh = frame.displayHeight;
  gl.viewport(0, 0, w, h);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, frame);
  gl.uniform2f(flat.size, fw, fh);
  gl.uniform2f(flat.centre, LENS.X * fw, LENS.Y * fh);
  gl.uniform1f(flat.radius, LENS.R * fh);
  gl.uniform1f(flat.k, LENS.K);
  // The wider side gets SPAN, the other its share, so the view keeps the canvas's shape
  gl.uniform2f(flat.extent, LENS.SPAN * Math.min(1, w / h), LENS.SPAN * Math.min(1, h / w));
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  frame.close();
  video.frames += 1;
  video.at = performance.now();
}

// A decoder error or a stream that stalls starts a new capture, which begins with a key frame
function openVideo() {
  closeVideo();
  if (!("VideoDecoder" in window)) {
    video.error = "This browser can't decode video (no WebCodecs), use Chrome or Edge.";
    return;
  }
  const ws = new WebSocket(`ws://${location.host}/ws/vr-video`);
  ws.binaryType = "arraybuffer";
  video.ws = ws;
  video.at = performance.now();
  let waitKey = true;
  ws.onmessage = (e) => {
    if (video.ws !== ws) return;
    if (typeof e.data === "string") {
      const msg = JSON.parse(e.data);
      video.error = msg.error || "";
      if (msg.codec) {
        if (video.decoder && video.decoder.state !== "closed") video.decoder.close();
        video.decoder = new VideoDecoder({
          output: drawVideoFrame,
          error: () => { if (video.ws === ws) setTimeout(openVideo, 500); },
        });
        video.decoder.configure({ codec: msg.codec, optimizeForLatency: true });
        video.codec = msg.codec;
        waitKey = true;
      }
      return;
    }
    const bytes = new Uint8Array(e.data);
    const key = bytes[0] === 1;
    if (!video.decoder || video.decoder.state !== "configured" || (waitKey && !key)) return;
    waitKey = false;
    // Frames behind in the queue are dropped rather than shown late
    if (video.decoder.decodeQueueSize > 2 && !key) return;
    video.decoder.decode(new EncodedVideoChunk({ type: key ? "key" : "delta", timestamp: performance.now() * 1000,
      data: bytes.subarray(1) }));
  };
  ws.onclose = () => { if (video.ws === ws && vr.open && vr.mode === "video") setTimeout(openVideo, 1000); };
}

function download(blob, ext) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `headset-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.${ext}`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// A clip of the video as the page draws it, until Stop: webm in Chrome and Edge, mp4 in Safari, which can't record
// webm
function toggleClip() {
  if (video.clip) {
    video.clip.stop();
    return;
  }
  const chunks = [];
  const type = MediaRecorder.isTypeSupported("video/webm") ? "video/webm" : "video/mp4";
  const rec = new MediaRecorder(videoCanvas.captureStream(30), { mimeType: type });
  rec.ondataavailable = (e) => chunks.push(e.data);
  rec.onstop = () => {
    video.clip = null;
    $("vr-clip").textContent = "Record a clip";
    download(new Blob(chunks, { type }), type === "video/webm" ? "webm" : "mp4");
  };
  rec.start(1000);
  video.clip = rec;
  $("vr-clip").textContent = "Stop and save the clip";
}

function videoStatus() {
  if (video.error) return video.error;
  if (!video.frames || performance.now() - video.at > 3000) return "Waiting for the headset's picture over USB.";
  return "The headset's own picture, left eye, about 0.1 s behind." + (vr.battery || "");
}

function tick() {
  if (!vr.open) return;
  if (vr.mode === "video") {
    $("vr-status").textContent = videoStatus();
    requestAnimationFrame(tick);
    return;
  }
  if (vr.mode === "orbit") orbit(camera, orbitView);
  if (performance.now() - vr.lastView > 2000 && vr.ws?.readyState === WebSocket.OPEN) {
    $("vr-status").textContent = "Waiting for the headset: plug the Quest in over USB and open BB Contact.";
  }
  render();
  requestAnimationFrame(tick);
}

function setMode(mode) {
  vr.mode = mode;
  canvas.hidden = mode === "video";
  videoCanvas.hidden = mode !== "video";
  $("vr-video-tools").hidden = mode !== "video";
  if (mode !== "video" && video.clip) video.clip.stop();
  if (mode === "video" && vr.open) {
    if (!video.ws) openVideo();
    return;
  }
  closeVideo();
  if (vr.open) resize();
  camera.fov = mode === "eyes" ? EYES_FOV : ORBIT_FOV;
  camera.up.set(0, 0, 1);
  camera.updateProjectionMatrix();
}

document.addEventListener("tabshown", (e) => {
  const on = e.detail === "vr";
  if (on === vr.open) return;
  vr.open = on;
  if (!on) {
    const ws = vr.ws;
    vr.ws = null;
    ws?.close();
    closeVideo();
    return;
  }
  resize();
  setMode(vr.mode);
  connect();
  requestAnimationFrame(tick);
});

document.querySelectorAll("[name=vr-camera]").forEach((r) => r.addEventListener("change", () => setMode(r.value)));
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("pointerdown", (e) => {
  if (e.button !== 2 || vr.mode !== "orbit") return;
  vr.turning = { x: e.clientX, y: e.clientY };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (!vr.turning) return;
  orbitView.yaw -= (e.clientX - vr.turning.x) * VIEW_DEG_PER_PX;
  orbitView.pitch = Math.min(85, Math.max(5, orbitView.pitch + (e.clientY - vr.turning.y) * VIEW_DEG_PER_PX));
  vr.turning = { x: e.clientX, y: e.clientY };
});
canvas.addEventListener("pointerup", () => { vr.turning = null; });
canvas.addEventListener("wheel", (e) => {
  if (vr.mode !== "orbit") return;
  e.preventDefault();
  orbitView.dist = Math.min(4, Math.max(0.3, orbitView.dist * Math.exp(e.deltaY * 0.001)));
}, { passive: false });
window.addEventListener("resize", () => { if (vr.open) resize(); });

buildParams($("vr-params"), CONTACT_PARAMS, "vr-", () => vr.params, () => {
  const { mass_scale, max_force, assist } = vr.params;
  postShowing("/api/contact/params", { mass_scale, max_force, assist }, "vr-error");
});
$("vr-spawn").innerHTML = Object.entries(OBJECT_NAMES)
  .map(([kind, name]) => `<button data-kind="${kind}">${name}</button>`).join("");
$("vr-spawn").addEventListener("click", (e) => {
  const kind = e.target.dataset?.kind;
  if (kind) postShowing("/api/vr/spawn", { kind }, "vr-error");
});
$("vr-keep-video").addEventListener("change", (e) =>
  postShowing("/api/vr/keep-video", { on: e.target.checked }, "vr-error"));
$("vr-snap").addEventListener("click", () => videoCanvas.toBlob((b) => b && download(b, "png")));
$("vr-clip").addEventListener("click", toggleClip);
$("vr-clear").addEventListener("click", () => postShowing("/api/vr/clear", null, "vr-error"));
