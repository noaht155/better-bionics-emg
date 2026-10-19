// Cursor tab: abstract control (cursor.py). The server maps the channel levels to the dot and runs the targets; the
// page draws them, shows the score and posts setting changes.

const CURSOR_PARAMS = [
  { key: "gain", label: "Gain", min: 0.2, max: 6, step: 0.1, help: "how far the dot moves per unit of level, higher reaches the targets with less effort" },
  { key: "smooth_s", label: "Smoothing s", min: 0, max: 1, step: 0.05, help: "how long the dot takes to follow, longer is steadier but slower" },
  { key: "rotation_deg", label: "Rotation", min: -180, max: 180, step: 5, help: "turns the whole map, so a movement that feels natural can point up" },
  { key: "flip", label: "Mirror", help: "mirrors the map, for a band turned the other way round" },
  { key: "targets", label: "Targets", min: 2, max: 12, step: 1, help: "how many target directions" },
  { key: "target_r", label: "Target distance", min: 0.2, max: 1, step: 0.05, help: "distance of the targets from the middle" },
  { key: "target_size", label: "Target size", min: 0.05, max: 0.5, step: 0.01, help: "radius of a target, smaller is harder" },
  { key: "home_size", label: "Home size", min: 0.05, max: 0.5, step: 0.01, help: "radius of the middle zone the dot has to rest in between targets" },
  { key: "home_s", label: "Home s", min: 0, max: 3, step: 0.1, help: "time in the middle before the next target shows" },
  { key: "dwell_s", label: "Dwell s", min: 0, max: 3, step: 0.1, help: "time the dot has to stay inside a target to hit it" },
  { key: "timeout_s", label: "Timeout s", min: 1, max: 30, step: 0.5, help: "time to reach a target before it counts as missed" },
];
const CURSOR_TRAIL = 20;
const cur = { params: null, trail: [] };

function sendCursorParams() {
  postShowing("/api/cursor/params", cur.params, "cur-error");
}

async function loadCursorSessions() {
  const sessions = await fetch("/api/sessions").then((r) => r.json());
  const usable = sessions.filter((s) => s.gestures > 0 && s.completed);
  fillSelect($("cur-session"), usable, (s) => `${s.name}, protocol ${s.protocol}${s.placement ? `, ${s.placement}` : ""}`);
}

$("cur-start").addEventListener("click", () => postShowing("/api/cursor/start", { session: $("cur-session").value }, "cur-error"));
$("cur-stop").addEventListener("click", () => post("/api/cursor/stop").catch(() => {}));
$("cur-targets").addEventListener("click", () => {
  postShowing("/api/cursor/targets", { on: !state.tools.cursor.targets_on }, "cur-error");
});

function drawCursorCanvas(c) {
  const { ctx, w, h } = sharpen($("cur-canvas"));
  const scale = (Math.min(w, h) / 2 - 10) / 1.2;
  const px = (x) => w / 2 + x * scale;
  const py = (y) => h / 2 - y * scale;
  const circle = (x, y, r) => { ctx.beginPath(); ctx.arc(px(x), py(y), r * scale, 0, 2 * Math.PI); };
  const p = cur.params;
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, w, h);
  ctx.lineWidth = 1;
  ctx.strokeStyle = COLORS.line;
  circle(0, 0, p.target_r); ctx.stroke();
  circle(0, 0, 1.2); ctx.stroke();

  // Each electrode's direction, and its level as a spoke, so the map can be learned by looking
  const n = c.levels.length;
  ctx.font = `12px ${MONO}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  c.levels.forEach((level, i) => {
    const a = (p.rotation_deg * Math.PI) / 180 + (p.flip ? -1 : 1) * (2 * Math.PI * i) / n;
    const [ux, uy] = [Math.cos(a), Math.sin(a)];
    ctx.strokeStyle = c.lost[i] ? COLORS.bad : COLORS.lineStrong;
    ctx.lineWidth = 6;
    ctx.beginPath(); ctx.moveTo(px(0), py(0)); ctx.lineTo(px(ux * level * p.target_r), py(uy * level * p.target_r)); ctx.stroke();
    ctx.fillStyle = c.lost[i] ? COLORS.bad : COLORS.muted;
    ctx.fillText(`ch ${i}`, px(ux * 1.12), py(uy * 1.12));
  });

  ctx.lineWidth = 2;
  const home = c.targets_on && c.phase === "home";
  ctx.strokeStyle = home ? COLORS.good : COLORS.muted;
  circle(0, 0, p.home_size); ctx.stroke();
  if (home && p.home_s > 0) {
    ctx.fillStyle = withAlpha(COLORS.good, 0.3);
    circle(0, 0, p.home_size * Math.min(1, c.t / p.home_s)); ctx.fill();
  }
  if (c.targets_on) {
    c.angles.forEach((deg, k) => {
      const a = (deg * Math.PI) / 180;
      const [x, y] = [p.target_r * Math.cos(a), p.target_r * Math.sin(a)];
      const active = c.phase === "go" && c.target === k;
      ctx.strokeStyle = active ? COLORS.accent : COLORS.line;
      circle(x, y, p.target_size); ctx.stroke();
      if (active) {
        ctx.fillStyle = withAlpha(COLORS.accent, 0.25);
        ctx.fill();
        if (c.dwell > 0 && p.dwell_s > 0) {
          ctx.fillStyle = withAlpha(COLORS.good, 0.6);
          circle(x, y, p.target_size * Math.min(1, c.dwell / p.dwell_s)); ctx.fill();
        }
      }
    });
  }

  ctx.strokeStyle = withAlpha(TRACKED_COLOR, 0.4);
  ctx.beginPath();
  cur.trail.forEach(([x, y], i) => (i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y))));
  ctx.stroke();
  ctx.fillStyle = TRACKED_COLOR;
  ctx.beginPath(); ctx.arc(px(c.x), py(c.y), 9, 0, 2 * Math.PI); ctx.fill();
}

function cursorScore(trials) {
  const hits = trials.filter((t) => t.hit);
  const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
  return { n: trials.length, rate: trials.length ? hits.length / trials.length : null, time: mean(hits.map((t) => t.time_s)) };
}

function drawCursor(c) {
  if (!c) return;
  if (!cur.params) {
    cur.params = { ...c.params };
    buildParams($("cur-params"), CURSOR_PARAMS, "cur-p-", () => cur.params, sendCursorParams);
    showParams(CURSOR_PARAMS, "cur-p-", cur.params);
  }
  const running = !!c.running;
  $("cur-stop").disabled = !running;
  $("cur-targets").disabled = !running;
  $("cur-targets").textContent = c.targets_on ? "Targets off" : "Targets on";
  $("cur-status").textContent = running ? `Running, levels from ${c.levels_from}${c.log ? `, logging to data/cursor/${c.log}` : ""}` : "Stopped";
  if (c.error) $("cur-error").textContent = c.error;
  if (currentTab === "cursor" && !running) drawIdle(sharpen($("cur-canvas")).ctx, "start to see the dot");
  if (currentTab !== "cursor" || !running || c.x == null) return;
  cur.trail.push([c.x, c.y]);
  cur.trail = cur.trail.slice(-CURSOR_TRAIL);
  drawCursorCanvas(c);

  const trials = c.trials || [];
  const all = cursorScore(trials);
  const last = cursorScore(trials.slice(-2 * c.angles.length));
  const pct = (v) => (v == null ? "no trials" : `${Math.round(100 * v)} %`);
  const sec = (v) => (v == null ? "no hits yet" : `${v.toFixed(1)} s`);
  $("cur-readout").innerHTML = `<div>${all.n}<small>trials</small></div><div>${pct(all.rate)}<small>hit</small></div>
    <div>${sec(all.time)}<small>time to hit</small></div><div>${pct(last.rate)}<small>hit, last ${2 * c.angles.length}</small></div>
    <div>${sec(last.time)}<small>time, last ${2 * c.angles.length}</small></div>`;
  const byAngle = {};
  for (const t of trials) (byAngle[t.angle] ||= []).push(t);
  $("cur-table").innerHTML = "<tr><th>Direction</th><th>Trials</th><th>Hit</th><th>Time to hit</th></tr>"
    + Object.keys(byAngle).sort((a, b) => a - b).map((a) => {
      const s = cursorScore(byAngle[a]);
      return `<tr><td>${a}°</td><td>${s.n}</td><td>${pct(s.rate)}</td><td>${sec(s.time)}</td></tr>`;
    }).join("");
}
