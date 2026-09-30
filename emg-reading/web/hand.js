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
