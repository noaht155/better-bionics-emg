// Rebuilds a hand skeleton from the 20 joint angles and draws it.
// Same angle definitions as hand_angles.py: x towards the thumb, y from the wrist to the fingers,
// z out of the palm, lengths in mm. Flexion bends towards +z, finger abduction turns towards +x.

const THUMB_REST_DEG = 45;

const FINGER_GEOMETRY = {
  index: { base: [22, 88, 0], dir: 8, len: [40, 24, 19] },
  middle: { base: [2, 92, 0], dir: 0, len: [45, 28, 20] },
  ring: { base: [-17, 86, 0], dir: -8, len: [42, 27, 20] },
  pinky: { base: [-33, 76, 0], dir: -16, len: [33, 19, 18] },
};
const THUMB_GEOMETRY = { base: [20, 22, 0], len: [42, 30, 25] };

const rad = (d) => (d * Math.PI) / 180;
const add = (a, b, s = 1) => [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const unit = (a) => { const n = Math.hypot(...a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };
const mix = (a, b, t) => add(a.map((v) => v * Math.cos(t)), b, Math.sin(t));

// Joints chain along a direction that turns from `ahead` towards `bend` by the running sum of the angles
function chain(start, ahead, bend, lengths, angles) {
  const points = [start];
  let total = 0;
  lengths.forEach((len, i) => {
    total += rad(angles[i]);
    points.push(add(points[points.length - 1], mix(ahead, bend, total), len));
  });
  return points;
}

// angles: {joint name: degrees}. Returns {name: [points]} with the wrist at the origin.
function buildHand(angles) {
  const a = (name) => angles[name] ?? 0;
  const z = [0, 0, 1];
  const hand = {};
  for (const [name, g] of Object.entries(FINGER_GEOMETRY)) {
    const phi = rad(g.dir + a(`${name}_mcp_abd`));
    const ahead = [Math.sin(phi), Math.cos(phi), 0];
    hand[name] = chain(g.base, ahead, z, g.len, [a(`${name}_mcp_flex`), a(`${name}_pip`), a(`${name}_dip`)]);
  }
  const phi = rad(THUMB_REST_DEG - a("thumb_cmc_flex"));
  const elev = rad(a("thumb_cmc_abd"));
  const meta = [Math.sin(phi) * Math.cos(elev), Math.cos(phi) * Math.cos(elev), Math.sin(elev)];
  // The thumb bends across the palm, towards the little finger and the palm side
  const across = unit([-1, 0, 1]);
  const bend = unit(add(across, meta, -dot(across, meta)));
  const cmc = THUMB_GEOMETRY.base;
  const mcp = add(cmc, meta, THUMB_GEOMETRY.len[0]);
  hand.thumb = [cmc, ...chain(mcp, meta, bend, THUMB_GEOMETRY.len.slice(1), [a("thumb_mcp"), a("thumb_ip")])];
  hand.palm = [[0, 0, 0], cmc, hand.index[0], hand.middle[0], hand.ring[0], hand.pinky[0], [0, 0, 0]];
  return hand;
}

// view: {yaw, pitch} in degrees. Orthographic, the palm faces the viewer at yaw 0.
function project(p, view, mirror) {
  const cy = Math.cos(rad(view.yaw)), sy = Math.sin(rad(view.yaw));
  const cp = Math.cos(rad(view.pitch)), sp = Math.sin(rad(view.pitch));
  let [x, y, zz] = p;
  [x, zz] = [x * cy + zz * sy, -x * sy + zz * cy];
  [y, zz] = [y * cp - zz * sp, y * sp + zz * cp];
  return { x: mirror ? -x : x, y: -y, depth: zz };
}

// Draws one hand. opts: {view, mirror, color, width, alpha}
function drawHand(ctx, angles, opts) {
  const hand = buildHand(angles);
  const { width: w, height: h } = ctx.canvas;
  const scale = Math.min(w, h) / 230;
  ctx.save();
  ctx.translate(w / 2, h / 2);
  ctx.scale(scale, scale);
  ctx.translate(0, 95);
  ctx.globalAlpha = opts.alpha ?? 1;
  ctx.strokeStyle = opts.color;
  ctx.fillStyle = opts.color;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = opts.width ?? 6;
  for (const points of Object.values(hand)) {
    const pts = points.map((p) => project(p, opts.view, opts.mirror));
    ctx.beginPath();
    pts.forEach((q, i) => (i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y)));
    ctx.stroke();
    for (const q of pts) {
      ctx.beginPath();
      ctx.arc(q.x, q.y, (opts.width ?? 6) * 0.7, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  ctx.restore();
}

// Finger thickness in mm per segment, base to tip. The thumb's first entry is its metacarpal
const THICKNESS = { thumb: [22, 19, 16], index: [17, 15, 13], middle: [18, 16, 14], ring: [17, 15, 13], pinky: [14, 12, 11] };

function shade(hex, t) {
  const n = parseInt(hex.slice(1), 16);
  const k = 0.55 + 0.45 * t;
  return `rgb(${Math.round((n >> 16) * k)}, ${Math.round(((n >> 8) & 255) * k)}, ${Math.round((n & 255) * k)})`;
}

const FOREARM_MM = 110;
// Half the palm's thickness, and the forearm's cross-section (half width across, half depth) at the wrist and the
// far end, in mm. Both are drawn as the outline of a solid, so they keep their thickness seen from the side
const PALM_HALF_MM = 5;
const FOREARM_SECTION = [[16, 10], [20, 13]];

// Outline (convex hull) of projected points, anticlockwise
function hull(pts) {
  const s = [...pts].sort((a, b) => a.x - b.x || a.y - b.y);
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const half = (list) => {
    const out = [];
    for (const q of list) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], q) <= 0) out.pop();
      out.push(q);
    }
    out.pop();
    return out;
  };
  return [...half(s), ...half([...s].reverse())];
}

// Bends the hand at the wrist (the origin) for the wrist cues: deviation about z, flexion about x towards the palm
// (+z), then the forearm's rotation about y
function turnAtWrist([x, y, z], angles) {
  const f = rad(angles.wrist_flex ?? 0), d = rad(angles.wrist_dev ?? 0), r = rad(angles.forearm_rot ?? 0);
  [x, y] = [x * Math.cos(d) + y * Math.sin(d), -x * Math.sin(d) + y * Math.cos(d)];
  [y, z] = [y * Math.cos(f) - z * Math.sin(f), y * Math.sin(f) + z * Math.cos(f)];
  [x, z] = [x * Math.cos(r) + z * Math.sin(r), -x * Math.sin(r) + z * Math.cos(r)];
  return [x, y, z];
}

// Draws a filled hand: palm and rounded finger segments, painted back to front and shaded by depth so the
// fingers in front cover the ones behind. opts: {view, mirror, color (#rrggbb), alpha}
function drawSolidHand(ctx, angles, opts) {
  const hand = buildHand(angles);
  const { width: w, height: h } = ctx.canvas;
  const arm = angles.arm ?? 0;
  // Smaller and moved up while the forearm shows, so it fits below the hand
  const scale = (Math.min(w, h) / 230) * (1 - 0.3 * arm);
  const view = { yaw: opts.view.yaw + (angles.view_yaw ?? 0), pitch: opts.view.pitch };
  const p = (q) => project(turnAtWrist(q, angles), view, opts.mirror);
  const parts = [];
  if (arm > 0.02) {
    // Two elliptical cross-sections, turned with the forearm. Always painted first, the hand sits in front of it
    const r = rad(angles.forearm_rot ?? 0);
    const ring = (y, [a, b]) => Array.from({ length: 16 }, (_, k) => {
      const t = (2 * Math.PI * k) / 16;
      const [x, z] = [-7 + a * Math.cos(t), b * Math.sin(t)];
      return project([x * Math.cos(r) + z * Math.sin(r), y, -x * Math.sin(r) + z * Math.cos(r)], view, opts.mirror);
    });
    const pts = [...ring(4, FOREARM_SECTION[0]), ...ring(-FOREARM_MM * arm, FOREARM_SECTION[1])];
    parts.push({ kind: "palm", pts: hull(pts), depth: -Infinity });
  }
  // Palm outline from the base of the thumb round the knuckles to both sides of the wrist, front and back face
  const outline = [hand.thumb[0], hand.index[0], hand.middle[0], hand.ring[0], hand.pinky[0], [-30, 8, 0], [16, 0, 0]];
  const palm = [-1, 1].flatMap((side) => outline.map(([x, y, z]) => p([x, y, z + side * PALM_HALF_MM])));
  parts.push({ kind: "palm", pts: hull(palm), depth: palm.reduce((a, q) => a + q.depth, 0) / palm.length - 20 });
  for (const [name, pts] of Object.entries(hand)) {
    if (name === "palm") continue;
    const proj = pts.map(p);
    for (let i = 0; i < proj.length - 1; i++) {
      parts.push({ kind: "bone", a: proj[i], b: proj[i + 1], width: THICKNESS[name][i], depth: (proj[i].depth + proj[i + 1].depth) / 2 });
    }
  }
  const depths = parts.map((x) => x.depth).filter(Number.isFinite);
  const lo = Math.min(...depths), hi = Math.max(...depths);
  parts.sort((x, y) => x.depth - y.depth);

  ctx.save();
  ctx.translate(w / 2, h / 2);
  ctx.scale(scale, scale);
  ctx.translate(0, 95 - 60 * arm);
  ctx.globalAlpha = opts.alpha ?? 1;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const edge = COLORS.bg;
  for (const part of parts) {
    const fill = shade(opts.color, Number.isFinite(part.depth) ? (part.depth - lo) / (hi - lo || 1) : 0.75);
    ctx.beginPath();
    if (part.kind === "palm") {
      part.pts.forEach((q, i) => (i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y)));
      ctx.closePath();
      // A thick stroke in the fill colour rounds off the palm's corners
      ctx.lineWidth = 18;
      ctx.strokeStyle = edge;
      ctx.stroke();
      ctx.lineWidth = 14;
      ctx.strokeStyle = fill;
      ctx.fillStyle = fill;
      ctx.stroke();
      ctx.fill();
    } else {
      ctx.moveTo(part.a.x, part.a.y);
      ctx.lineTo(part.b.x, part.b.y);
      ctx.lineWidth = part.width + 3;
      ctx.strokeStyle = edge;
      ctx.stroke();
      ctx.lineWidth = part.width;
      ctx.strokeStyle = fill;
      ctx.stroke();
    }
  }
  ctx.restore();
}

// Lets the user drag on a canvas to turn the view
function attachRotate(canvas, view, onChange) {
  let last = null;
  canvas.addEventListener("pointerdown", (e) => { last = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointerup", () => { last = null; });
  canvas.addEventListener("pointermove", (e) => {
    if (!last) return;
    view.yaw += (e.clientX - last[0]) * 0.5;
    view.pitch = Math.max(-89, Math.min(89, view.pitch + (e.clientY - last[1]) * 0.5));
    last = [e.clientX, e.clientY];
    onChange();
  });
}
