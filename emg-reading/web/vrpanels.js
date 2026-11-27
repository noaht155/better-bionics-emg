// The headset's panels (menu, cue card, Menu button) on the VR tab, from what Ui.Describe sends: drawn into the 3D
// view where they float, with the pointing ray, and as flat copies beside it. Clicking a control on a copy makes it
// pulse in the headset (/api/vr/point), to show the user where to press.
import { THREE, color } from "./scene3d.js";

// Room under a panel for its move bar
const BAR_M = 0.05;
const PX_PER_M = 1400;
const MAX_COPY_PX_PER_M = 1300;
const PULSE_MS = 3000;
const PANEL_NAMES = {
  menu: "Menu", card: "Cue card", dock: "Menu button", tour: "Tour", board: "Model output", finish: "Finish card",
  controls: "Controls card",
};

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const C = {
  bg: css("--bg"), panel: css("--panel"), raised: css("--raised"), line: css("--line"), lineStrong: css("--line-strong"),
  text: css("--text"), muted: css("--muted"), accent: css("--accent"),
};
const SANS = css("--font-sans"), MONO = css("--font-mono");

// Clicked on the laptop, name -> when, so the copy pulses as the headset does
const pointed = new Map();

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
}

// A panel's text as the headset shows it, without TextMeshPro's tags, wrapped to its box's width
function drawText(ctx, it, X, Y, s) {
  ctx.font = `${it.on ? "600 " : ""}${it.s * s}px ${SANS}`;
  ctx.fillStyle = it.c || C.text;
  ctx.textBaseline = "top";
  ctx.textAlign = it.a === 0.5 ? "center" : it.a === 1 ? "right" : "left";
  const x = it.a === 0.5 ? X(it.x + it.w / 2) : it.a === 1 ? X(it.x + it.w) : X(it.x);
  const lineH = it.s * 1.35 * s, maxW = it.w * s, bottom = Y(it.b) + lineH;
  let y = Y(it.y);
  for (const para of it.t.replace(/<[^>]*>/g, "").split("\n")) {
    let line = "";
    for (const word of para.split(" ")) {
      const next = line ? `${line} ${word}` : word;
      if (line && ctx.measureText(next).width > maxW) {
        ctx.fillText(line, x, y);
        y += lineH;
        line = word;
      } else {
        line = next;
      }
    }
    if (y > bottom) return;
    ctx.fillText(line, x, y);
    y += lineH;
  }
}

function fitText(ctx, text, max) {
  if (ctx.measureText(text).width <= max) return text;
  let t = text;
  while (t.length > 1 && ctx.measureText(`${t}…`).width > max) t = t.slice(0, -1);
  return `${t}…`;
}

// One panel at s px per metre, its top centre at the canvas' top middle
function drawPanel(ctx, p, s, now) {
  const H = p.h + BAR_M;
  ctx.clearRect(0, 0, p.w * s, H * s);
  roundRect(ctx, 0.5, 0.5, p.w * s - 1, p.h * s - 1, 0.012 * s);
  ctx.fillStyle = C.panel;
  ctx.fill();
  ctx.strokeStyle = C.lineStrong;
  ctx.lineWidth = 1;
  ctx.stroke();
  const X = (x) => (p.w / 2 + x) * s, Y = (y) => -y * s;
  p.items.forEach((it, i) => {
    const hover = i === p.hover, press = i === p.press;
    const since = now - (pointed.get(`${p.name}\t${p.page}\t${it.n}`) ?? -Infinity);
    const pulse = since < PULSE_MS && Math.cos((since / 1000) * 3 * Math.PI) < 0;
    const lit = hover || pulse;
    const left = X(it.x - it.w / 2), top = Y(it.y + it.h / 2), w = it.w * s, h = it.h * s;
    ctx.textBaseline = "middle";
    switch (it.k) {
      case "section":
        ctx.font = `${0.0105 * s}px ${MONO}`;
        ctx.fillStyle = C.muted;
        ctx.textAlign = "left";
        ctx.fillText(it.n, X(it.x), Y(it.y));
        break;
      case "tab":
      case "close":
        ctx.font = `${(it.k === "tab" ? 0.011 : 0.02) * s}px ${it.k === "tab" ? MONO : SANS}`;
        ctx.fillStyle = lit ? C.accent : it.on ? C.text : C.muted;
        ctx.textAlign = "center";
        ctx.fillText(it.k === "tab" ? it.n.toUpperCase() : "×", X(it.x), Y(it.y));
        if (it.on) {
          ctx.fillStyle = C.accent;
          ctx.fillRect(X(it.x) - w * 0.35, Y(it.y) + 0.018 * s, w * 0.7, 0.0025 * s);
        }
        break;
      case "button":
      case "primary":
        roundRect(ctx, left, top, w, h, 0.003 * s);
        ctx.fillStyle = it.k === "primary" ? (press || pulse ? C.accent : C.text) : lit ? C.raised : C.bg;
        ctx.fill();
        ctx.strokeStyle = lit ? C.accent : C.lineStrong;
        ctx.lineWidth = lit ? 2 : 1;
        ctx.stroke();
        ctx.font = `${it.k === "primary" ? "700 " : ""}${0.0135 * s}px ${SANS}`;
        ctx.fillStyle = it.k === "primary" ? C.bg : C.text;
        ctx.textAlign = "center";
        ctx.fillText(fitText(ctx, it.n, w - 6), left + w / 2, top + h / 2);
        break;
      case "text":
        drawText(ctx, it, X, Y, s);
        break;
      case "bar":
        ctx.fillStyle = C.raised;
        ctx.fillRect(left, Y(it.y) - h / 2, w, h);
        ctx.fillStyle = C.accent;
        ctx.fillRect(left, Y(it.y) - h / 2, w * it.v, h);
        break;
      case "toggle":
        ctx.font = `${0.015 * s}px ${SANS}`;
        ctx.fillStyle = C.text;
        ctx.textAlign = "left";
        ctx.fillText(fitText(ctx, it.n, left - X(it.lx) - 8), X(it.lx), Y(it.y));
        roundRect(ctx, left, top, w, h, h / 2);
        ctx.fillStyle = it.on ? C.accent : C.raised;
        ctx.fill();
        if (lit) {
          ctx.strokeStyle = C.accent;
          ctx.lineWidth = 2;
          ctx.stroke();
        }
        ctx.beginPath();
        ctx.arc(it.on ? left + w - h / 2 : left + h / 2, top + h / 2, 0.008 * s, 0, 2 * Math.PI);
        ctx.fillStyle = C.text;
        ctx.fill();
        break;
      case "slider": {
        ctx.font = `${0.015 * s}px ${SANS}`;
        ctx.fillStyle = C.text;
        ctx.textAlign = "left";
        ctx.fillText(fitText(ctx, it.n, left - X(it.lx) - 8), X(it.lx), Y(it.y));
        ctx.font = `${0.013 * s}px ${MONO}`;
        ctx.textAlign = "right";
        ctx.fillText(it.t || "", left + w + 0.045 * s + 0.015 * s, Y(it.y));
        const track = 0.006 * s, ty = Y(it.y) - track / 2;
        ctx.fillStyle = C.raised;
        ctx.fillRect(left, ty, w, track);
        ctx.fillStyle = C.accent;
        ctx.fillRect(left, ty, w * it.v, track);
        ctx.beginPath();
        ctx.arc(left + w * it.v, Y(it.y), 0.008 * s * (lit ? 1.25 : 1), 0, 2 * Math.PI);
        ctx.fillStyle = lit ? C.accent : C.text;
        ctx.fill();
        break;
      }
      case "handle":
        roundRect(ctx, X(it.x) - 0.04 * s, Y(it.y) - 0.004 * s, 0.08 * s, 0.008 * s, 0.004 * s);
        ctx.fillStyle = lit || press ? C.accent : C.lineStrong;
        ctx.fill();
        break;
    }
    // A press through the ray has to be held: its bar fills under the control
    if (press && p.hold > 0 && it.k !== "handle") {
      ctx.fillStyle = C.accent;
      ctx.fillRect(it.k === "toggle" ? X(it.lx) : left, top + h + 2, (it.k === "toggle" ? left + w - X(it.lx) : w) * p.hold, 3);
    }
  });
}

function pulsing(p, now) {
  return p.items.some((it) => now - (pointed.get(`${p.name}	${p.page}	${it.n}`) ?? -Infinity) < PULSE_MS);
}

function itemAt(p, x, y) {
  // x, y in metres from the top centre, y down
  return p.items.find((it) => it.k !== "section" && Math.abs(x - it.x) <= it.w / 2 + 0.004
    && Math.abs(-y - it.y) <= it.h / 2 + 0.004);
}

// Flat copies in a container, one canvas per panel, clickable
export function createCopies(container, errorId, post) {
  const copies = {};
  function copyFor(name) {
    if (copies[name]) return copies[name];
    const box = document.createElement("div");
    box.className = "vr-panel";
    const label = document.createElement("div");
    label.className = "muted";
    label.textContent = PANEL_NAMES[name];
    const canvas = document.createElement("canvas");
    box.append(label, canvas);
    container.append(box);
    const copy = { box, canvas, panel: null };
    canvas.addEventListener("click", (e) => {
      const p = copy.panel;
      if (!p) return;
      const r = canvas.getBoundingClientRect();
      const s = r.width / p.w;
      const it = itemAt(p, (e.clientX - r.left) / s - p.w / 2, (e.clientY - r.top) / s);
      if (!it) return;
      pointed.set(`${p.name}\t${p.page}\t${it.n}`, performance.now());
      post("/api/vr/point", { panel: p.name, page: p.page, name: it.n }, errorId);
    });
    canvas.addEventListener("mousemove", (e) => {
      const p = copy.panel;
      if (!p) return;
      const r = canvas.getBoundingClientRect();
      const s = r.width / p.w;
      canvas.style.cursor = itemAt(p, (e.clientX - r.left) / s - p.w / 2, (e.clientY - r.top) / s) ? "pointer" : "";
    });
    return (copies[name] = copy);
  }

  return function show(panels) {
    const now = performance.now();
    const seen = new Set();
    for (const p of panels || []) {
      seen.add(p.name);
      const copy = copyFor(p.name);
      copy.panel = p;
      copy.box.hidden = false;
      // Drawn at most at the size the menu takes up at full width, so the small panels don't blow up
      const cssW = Math.min(container.clientWidth, p.w * MAX_COPY_PX_PER_M);
      if (!cssW) continue;
      copy.canvas.style.width = `${cssW}px`;
      const s = cssW / p.w, dpr = window.devicePixelRatio || 1;
      const cssH = (p.h + BAR_M) * s;
      const bw = Math.round(cssW * dpr), bh = Math.round(cssH * dpr);
      if (copy.canvas.width !== bw || copy.canvas.height !== bh) {
        Object.assign(copy.canvas, { width: bw, height: bh });
        copy.canvas.style.height = `${cssH}px`;
      }
      const ctx = copy.canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawPanel(ctx, p, s, now);
    }
    for (const [name, copy] of Object.entries(copies)) if (!seen.has(name)) copy.box.hidden = true;
    return seen.size;
  };
}

// The panels where they float in the 3D view, and the ray
export function createOverlay(world) {
  const group = new THREE.Group();
  world.add(group);
  const planes = {};
  const rayGeo = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(6), 3));
  const ray = new THREE.Line(rayGeo, new THREE.LineBasicMaterial({ color: color("--accent") }));
  ray.frustumCulled = false;
  group.add(ray);
  const v = (a) => new THREE.Vector3(a[0], a[1], a[2]);

  function planeFor(name) {
    if (planes[name]) return planes[name];
    const canvas = document.createElement("canvas");
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({ map: texture, transparent: true, side: THREE.DoubleSide, depthWrite: false }));
    mesh.matrixAutoUpdate = false;
    group.add(mesh);
    return (planes[name] = { canvas, texture, mesh, key: "" });
  }

  return function show(panels, rayPoints, now = performance.now()) {
    const seen = new Set();
    for (const p of panels || []) {
      seen.add(p.name);
      const plane = planeFor(p.name);
      plane.mesh.visible = true;
      const H = p.h + BAR_M;
      const key = JSON.stringify(p);
      if (key !== plane.key || pulsing(p, now)) {
        plane.key = key;
        const w = Math.round(p.w * PX_PER_M), h = Math.round(H * PX_PER_M);
        if (plane.canvas.width !== w || plane.canvas.height !== h) Object.assign(plane.canvas, { width: w, height: h });
        drawPanel(plane.canvas.getContext("2d"), p, PX_PER_M, now);
        plane.texture.needsUpdate = true;
      }
      const right = v(p.right), up = v(p.up);
      const normal = new THREE.Vector3().crossVectors(right, up).normalize();
      const centre = v(p.pos).addScaledVector(up, -H / 2);
      plane.mesh.matrix.makeBasis(right.multiplyScalar(p.w), up.multiplyScalar(H), normal).setPosition(centre);
    }
    for (const [name, plane] of Object.entries(planes)) if (!seen.has(name)) plane.mesh.visible = false;
    ray.visible = rayPoints?.length === 6;
    if (ray.visible) {
      rayGeo.attributes.position.array.set(rayPoints);
      rayGeo.attributes.position.needsUpdate = true;
    }
  };
}
