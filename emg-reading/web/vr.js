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
// flattened back on the GPU: an output point u (in lens radii, R half the frame's height, from the lens centre at
// x, y of the frame, y up) comes from the recording at centre + u (1 + k1 r^2 + k2 r^4) R. box is the part of u shown
// (left, right, bottom, top): out to the eye picture's edges (eyeBox), letterboxed in the canvas. measureLens fits the
// warp from the headset's dot grid and saves it; until then Noah's Quest 3 as measured on 2026-11-27 (161 dots, rows
// 7.4 % of their spacing off straight before, 2.4 % after)
let lens = { k1: -0.221, k2: 0.031, x: 0.2868, y: 0.5076, box: [-1.535, 0.882, -1.438, 1.538] };
const LENS_R = 0.5;
fetch("/api/vr/lens").then((r) => r.json()).then((l) => { if (l.k1 != null) lens = l; }).catch(() => {});
const FLAT_SHADER = `precision mediump float;
  varying vec2 p; uniform sampler2D tex; uniform vec2 size, centre; uniform vec4 box; uniform float radius, k1, k2;
  void main() {
    vec2 u = vec2(mix(box.x, box.y, p.x * 0.5 + 0.5), mix(box.z, box.w, p.y * 0.5 + 0.5));
    float r2 = dot(u, u);
    vec2 s = centre + vec2(u.x, -u.y) * (1.0 + k1 * r2 + k2 * r2 * r2) * radius;
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
  return { gl, size: at("size"), centre: at("centre"), box: at("box"), radius: at("radius"), k1: at("k1"), k2: at("k2") };
}

function drawVideoFrame(frame) {
  const w = videoCanvas.clientWidth, h = videoCanvas.clientHeight;
  if (videoCanvas.width !== w || videoCanvas.height !== h) Object.assign(videoCanvas, { width: w, height: h });
  const fw = frame.displayWidth, fh = frame.displayHeight;
  // The frame goes through a 2D canvas first: a decoded frame handed straight to WebGL came out black on Noah's
  // browser (2026-11-27), drawing it on a 2D canvas is what worked before
  flat ??= { copy: document.createElement("canvas") };
  if (flat.copy.width !== fw || flat.copy.height !== fh) Object.assign(flat.copy, { width: fw, height: fh });
  flat.copy.getContext("2d").drawImage(frame, 0, 0);
  frame.close();
  // Kept so a picture or a clip can be saved from the canvas
  if (!flat.gl) Object.assign(flat, flatView(videoCanvas.getContext("webgl", { preserveDrawingBuffer: true })));
  const { gl } = flat;
  gl.viewport(0, 0, w, h);
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  const [left, right, bottom, top] = lens.box;
  const aspect = (right - left) / (top - bottom);
  const vw = Math.min(w, h * aspect), vh = vw / aspect;
  gl.viewport((w - vw) / 2, (h - vh) / 2, vw, vh);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, flat.copy);
  gl.uniform2f(flat.size, fw, fh);
  gl.uniform2f(flat.centre, lens.x * fw, lens.y * fh);
  gl.uniform1f(flat.radius, LENS_R * fh);
  gl.uniform1f(flat.k1, lens.k1);
  gl.uniform1f(flat.k2, lens.k2);
  gl.uniform4f(flat.box, left, right, bottom, top);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  video.frames += 1;
  video.at = performance.now();
}

// Measuring the lens warp: the headset shows white dots on a flat square in front of the eyes (App.ShowGrid), so in
// the picture each eye gets they lie in straight rows and columns. The dots are found in the left eye of one frame
// and the warp fitted (Nelder-Mead) until every row and column of them is straight again. box is then the part the
// dots cover: all of the eye the recording shows
function findDots(img, w, h) {
  const seen = new Uint8Array(w * h), dots = [];
  const bright = (i) => img[4 * i] + img[4 * i + 1] + img[4 * i + 2] > 450;
  // The left eye only, the right half is the other eye
  for (let y = 0; y < h; y++) for (let x = 0; x < w / 2; x++) {
    const start = y * w + x;
    if (seen[start] || !bright(start)) continue;
    let n = 0, sx = 0, sy = 0;
    const stack = [start];
    seen[start] = 1;
    while (stack.length) {
      const i = stack.pop(), px = i % w, py = (i - px) / w;
      n++;
      sx += px;
      sy += py;
      for (const j of [i - 1, i + 1, i - w, i + w]) {
        const jx = j % w;
        if (j < 0 || j >= w * h || Math.abs(jx - px) > 1 || jx >= w / 2 || seen[j] || !bright(j)) continue;
        seen[j] = 1;
        stack.push(j);
      }
    }
    if (n >= 3 && n <= 2500) dots.push([sx / n, sy / n]);
  }
  return dots;
}

// Recording pixels to u for lens l: the warp undone by fixed-point steps
function toU(l, d, w, h) {
  const R = LENS_R * h, ux0 = (d[0] - l.x * w) / R, uy0 = -(d[1] - l.y * h) / R;
  let ux = ux0, uy = uy0;
  for (let i = 0; i < 20; i++) {
    const r2 = ux * ux + uy * uy, f = Math.max(0.2, 1 + l.k1 * r2 + l.k2 * r2 * r2);
    ux = ux0 / f;
    uy = uy0 / f;
  }
  return [ux, uy];
}

// Rows and columns of the dots (lists of indices), traced in the recording itself: each dot's nearest neighbour to the
// right and below, kept only when that dot has it as its own nearest to the left and above. Locally the warp is
// small, so this follows the curved rows without knowing the warp. step: the median distance to a neighbour (px)
function groupDots(dots) {
  const near = (i, dir) => {
    let best = -1, bestD = Infinity;
    const [x, y] = dots[i];
    dots.forEach(([qx, qy], j) => {
      const dx = qx - x, dy = qy - y, along = dir === 0 ? dx : dir === 1 ? dy : dir === 2 ? -dx : -dy;
      const across = dir % 2 === 0 ? Math.abs(dy) : Math.abs(dx);
      const d = Math.hypot(dx, dy);
      if (j !== i && along > 0 && across < along && d < bestD) { best = j; bestD = d; }
    });
    return [best, bestD];
  };
  const next = [[], []], has = [[], []], gaps = [];
  dots.forEach((_, i) => {
    for (const axis of [0, 1]) {
      const [j, d] = near(i, axis);
      if (j >= 0 && near(j, axis + 2)[0] === i) {
        next[axis][i] = j;
        has[axis][j] = true;
        gaps.push(d);
      }
    }
  });
  const lines = [];
  for (const axis of [0, 1]) dots.forEach((_, i) => {
    if (has[axis][i]) return;
    // A sharp turn or a step much longer or shorter than the last is a jump to another row: the line ends there
    let line = [i];
    for (let j = next[axis][i]; j !== undefined; j = next[axis][j]) {
      if (line.length >= 2) {
        const [a, b] = [dots[line[line.length - 2]], dots[line[line.length - 1]]], c = dots[j];
        const d1 = [b[0] - a[0], b[1] - a[1]], d2 = [c[0] - b[0], c[1] - b[1]];
        const l1 = Math.hypot(...d1), l2 = Math.hypot(...d2);
        const turn = Math.acos(Math.min(1, (d1[0] * d2[0] + d1[1] * d2[1]) / (l1 * l2)));
        if (turn > 0.35 || l2 / l1 > 1.5 || l2 / l1 < 0.67) {
          if (line.length >= 5) lines.push(line);
          line = [line[line.length - 1]];
        }
      }
      line.push(j);
    }
    if (line.length >= 5) lines.push(line);
  });
  gaps.sort((a, b) => a - b);
  return { lines, step: gaps[Math.floor(gaps.length / 2)] || 1 };
}

// How far the dots sit off a straight line through each row and column, as a share of the spacing between them in u
// (so shrinking the picture doesn't look straighter)
function bend(l, dots, lines, w, h) {
  const u = dots.map((d) => toU(l, d, w, h));
  let sum = 0, n = 0, gap = 0, gaps = 0;
  for (const line of lines) {
    const p = line.map((i) => u[i]);
    for (let k = 1; k < p.length; k++) { gap += Math.hypot(p[k][0] - p[k - 1][0], p[k][1] - p[k - 1][1]); gaps++; }
    const mx = p.reduce((a, q) => a + q[0], 0) / p.length, my = p.reduce((a, q) => a + q[1], 0) / p.length;
    let xx = 0, yy = 0, xy = 0;
    for (const [x, y] of p) {
      xx += (x - mx) ** 2;
      yy += (y - my) ** 2;
      xy += (x - mx) * (y - my);
    }
    // The smaller eigenvalue of the spread: the squared distances off the best line
    const small = (xx + yy) / 2 - Math.sqrt(((xx - yy) / 2) ** 2 + xy * xy);
    sum += Math.max(small, 0);
    n += p.length;
  }
  return Math.sqrt(sum / Math.max(n, 1)) / (gap / Math.max(gaps, 1));
}

function nelderMead(f, x0, scale, steps = 400) {
  const shrinkTo = (best, x) => x.map((v, j) => best[j] + 0.5 * (v - best[j]));
  let pts = [x0, ...x0.map((_, i) => x0.map((v, j) => v + (i === j ? scale[i] : 0)))].map((x) => [x, f(x)]);
  for (let s = 0; s < steps; s++) {
    pts.sort((a, b) => a[1] - b[1]);
    const n = x0.length, best = pts[0], worst = pts[n];
    const c = x0.map((_, j) => pts.slice(0, n).reduce((a, p) => a + p[0][j], 0) / n);
    const at = (t) => c.map((v, j) => v + t * (worst[0][j] - v));
    const r = at(-1), fr = f(r);
    if (fr < best[1]) {
      const e = at(-2), fe = f(e);
      pts[n] = fe < fr ? [e, fe] : [r, fr];
    } else if (fr < pts[n - 1][1]) {
      pts[n] = [r, fr];
    } else {
      const k = at(0.5), fk = f(k);
      if (fk < worst[1]) pts[n] = [k, fk];
      else pts = pts.map((p, i) => (i === 0 ? p : [shrinkTo(best[0], p[0]), f(shrinkTo(best[0], p[0]))]));
    }
  }
  pts.sort((a, b) => a[1] - b[1]);
  return pts[0];
}

// The view out to where the left eye's picture ends, through the lens centre: its left edge, the middle of the
// frame (the other eye starts there), the top and the bottom. Not the dots' extent: the grid doesn't fill the view
// and the picture came out cropped (Noah, 2026-11-27). The corners past the round edge stay black
function eyeBox(l, w, h) {
  return [toU(l, [6, l.y * h], w, h)[0], toU(l, [w / 2 - 4, l.y * h], w, h)[0],
    toU(l, [l.x * w, h - 4], w, h)[1], toU(l, [l.x * w, 4], w, h)[1]];
}

async function measureLens() {
  const say = (text) => { $("vr-lens-status").textContent = text; };
  $("vr-error").textContent = "";
  if (!flat?.copy?.width) {
    say("Turn on Headset video first, with the headset on.");
    return;
  }
  $("vr-lens").disabled = true;
  try {
    say("Showing the grid in the headset, keep your head still");
    if (!(await postShowing("/api/vr/grid", { on: true }, "vr-error"))) return;
    await new Promise((r) => setTimeout(r, 2000));
    const w = flat.copy.width, h = flat.copy.height;
    const img = flat.copy.getContext("2d").getImageData(0, 0, w, h).data;
    await postShowing("/api/vr/grid", { on: false }, "vr-error");
    const dots = findDots(img, w, h);
    if (dots.length < 40) {
      say(`Found only ${dots.length} dots: is the headset on, Headset video running and the app in front?`);
      return;
    }
    let l = { ...lens };
    const grouped = groupDots(dots);
    if (grouped.lines.length < 8) {
      say(`Found ${dots.length} dots but only ${grouped.lines.length} rows and columns: keep your hands out of the view`);
      return;
    }
    // The centre stays within the left eye's half
    const cost = (v) => (v[2] < 0.1 || v[2] > 0.4 || v[3] < 0.3 || v[3] > 0.7 ? 1e9
      : bend({ k1: v[0], k2: v[1], x: v[2], y: v[3] }, dots, grouped.lines, w, h));
    const before = cost([l.k1, l.k2, l.x, l.y]);
    // From where the last fit ended, twice, so a stall on the first pass gets another start
    for (let round = 0; round < 2; round++) {
      const [v] = nelderMead(cost, [l.k1, l.k2, l.x, l.y], [0.05, 0.05, 0.01, 0.01]);
      l = { ...l, k1: v[0], k2: v[1], x: v[2], y: v[3] };
    }
    const after = cost([l.k1, l.k2, l.x, l.y]);
    l.box = eyeBox(l, w, h);
    l.lines = grouped.lines.length;
    l.error = after;
    if (!(await postShowing("/api/vr/lens", l, "vr-error"))) return;
    lens = l;
    say(`Measured from ${dots.length} dots in ${grouped.lines.length} rows and columns: they sat ${pct(before)} of the `
      + `spacing between dots off straight before, ${pct(after)} now. Saved, used from now on.`);
  } finally {
    $("vr-lens").disabled = false;
  }
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
// A reload keeps the switch where it was without a change event (Chrome restores form state): the view follows it,
// or the switch shows Headset video over the 3D view (Noah, 2026-11-27)
setMode(document.querySelector("[name=vr-camera]:checked")?.value ?? "eyes");
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
$("vr-lens").addEventListener("click", measureLens);
$("vr-snap").addEventListener("click", () => videoCanvas.toBlob((b) => b && download(b, "png")));
$("vr-clip").addEventListener("click", toggleClip);
$("vr-clear").addEventListener("click", () => postShowing("/api/vr/clear", null, "vr-error"));
