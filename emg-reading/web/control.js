// Control tab: the gesture model with the control layer (control.py) on its output. The server steps the
// controller inside the model's loop and sends its state with the rest; the page draws it and posts setting changes.

const CONTROL_PARAMS = [
  { key: "threshold", label: "Threshold", min: 0, max: 0.99, step: 0.01, help: "grip probability that counts as a vote, also for the open-hand release" },
  { key: "vote_n", label: "Votes", min: 1, max: 15, step: 1, help: "steps in a row (50 ms each) the same grip has to win to start" },
  { key: "start_level", label: "Start level", min: 0.01, max: 0.6, step: 0.01, help: "effort that counts as a rise from rest" },
  { key: "onset_s", label: "Onset s", min: 0.1, max: 5, step: 0.05, help: "time after crossing the start level in which a grip can still be picked" },
  { key: "closure_mode", label: "Closure", options: ["position", "speed"], help: "position: effort sets how far the hand is closed. speed: effort above the hold level closes, below opens" },
  { key: "gain", label: "Gain", min: 0, max: 10, step: 0.1, help: "closure per unit of effort (position) or per second (speed)" },
  { key: "hold_level", label: "Hold level", min: 0, max: 0.6, step: 0.01, help: "speed mode: effort at which the closure stays still" },
  { key: "hold_on_relax", label: "Hold on relax", help: "relaxing keeps the hand where it is instead of opening it" },
  { key: "release_mode", label: "Release", options: ["rest_or_class", "either", "open_class", "rest"], help: "rest_or_class: rest or the open hand wins, or effort stays low. open_class: the open hand gesture. rest: effort stays low. either: open hand or low effort" },
  { key: "release_n", label: "Release votes", min: 1, max: 10, step: 1, help: "steps in a row the open hand (or rest, in rest_or_class) has to win to release" },
  { key: "release_level", label: "Release level", min: 0, max: 0.5, step: 0.01, help: "effort below which the hand counts as at rest" },
  { key: "release_s", label: "Release s", min: 0, max: 3, step: 0.05, help: "time below the release level before the grip lets go" },
  { key: "max_distance", label: "Max distance", min: 0, max: 15, step: 0.5, help: "windows farther than this from every trained grip count as rest, 0 turns it off. Needs a model trained since 2026-10-15" },
  { key: "spring_w", label: "Spring", min: 0, max: 30, step: 0.5, help: "how fast the hand follows its target, 0 jumps straight there" },
];
const TRACE_STEPS = 200;
const ctlView = { yaw: 25, pitch: -20 };
const ctl = { params: null, trace: [], lastStep: null, starts: 0, releases: 0 };

function buildControlParams() {
  const box = $("ctl-params");
  box.innerHTML = CONTROL_PARAMS.map((d) => {
    let input;
    if (d.options) input = `<select id="ctl-${d.key}">${d.options.map((o) => `<option>${o}</option>`).join("")}</select><output></output>`;
    else if (d.min == null) input = `<input type="checkbox" id="ctl-${d.key}"><output></output>`;
    else input = `<input type="range" id="ctl-${d.key}" min="${d.min}" max="${d.max}" step="${d.step}"><output></output>`;
    return `<label class="param"><span>${d.label}</span>${input}<small>${d.help}</small></label>`;
  }).join("");
  for (const d of CONTROL_PARAMS) {
    $(`ctl-${d.key}`).addEventListener("input", (e) => {
      const el = e.target;
      ctl.params[d.key] = d.options ? el.value : d.min == null ? el.checked : Number(el.value);
      showControlParams();
      sendControlParams();
    });
  }
}

function showControlParams() {
  for (const d of CONTROL_PARAMS) {
    const el = $(`ctl-${d.key}`);
    const v = ctl.params[d.key];
    if (d.min == null && !d.options) el.checked = v;
    else el.value = v;
    if (!d.options && d.min != null) el.nextElementSibling.textContent = v;
  }
}

function sendControlParams() {
  $("ctl-error").textContent = "";
  post("/api/control/params", ctl.params).catch((err) => { $("ctl-error").textContent = err.message; });
}

async function loadControlLists() {
  const [models, presets] = await Promise.all([fetch("/api/models"), fetch("/api/control/presets")].map((r) => r.then((x) => x.json())));
  fillSelect($("ctl-model"), models, (m) => {
    const acc = m.balanced != null ? `, ${Math.round(100 * m.balanced)} % balanced on unseen sessions` : "";
    return `${m.name}, trained on ${m.sessions.map((s) => s.slice(0, 17).replace("_", " ")).join(", ")}${acc}`;
  });
  const running = state?.tools?.predictor?.running && state.tools.predictor.control && state.tools.predictor.model;
  if (running && models.some((m) => m.name === running)) $("ctl-model").value = running;
  ctl.presets = presets;
  const select = $("ctl-preset");
  const current = select.value;
  select.innerHTML = Object.keys(presets).map((name) => `<option>${name}</option>`).join("");
  if (current in presets) select.value = current;
}

$("ctl-start").addEventListener("click", async () => {
  $("ctl-error").textContent = "";
  $("ctl-haptics").textContent = "";
  try {
    const r = await post("/api/control/start", { name: $("ctl-model").value, ...linkBody() });
    $("ctl-haptics").textContent = r.haptics || "";
  } catch (err) {
    $("ctl-error").textContent = err.message;
  }
});
$("ctl-stop").addEventListener("click", () => post("/api/model/stop").catch(() => {}));
$("ctl-reset").addEventListener("click", () => post("/api/control/reset").catch(() => {}));

$("ctl-load").addEventListener("click", () => {
  const preset = ctl.presets?.[$("ctl-preset").value];
  if (!preset || !ctl.params) return;
  ctl.params = { ...ctl.params, ...preset };
  showControlParams();
  sendControlParams();
  $("ctl-preset-note").textContent = `loaded ${$("ctl-preset").value}`;
});

$("ctl-save").addEventListener("click", async () => {
  const name = $("ctl-preset-name").value.trim();
  try {
    await post("/api/control/presets", { name, params: ctl.params });
    $("ctl-preset-note").textContent = `saved ${name}`;
    await loadControlLists();
    $("ctl-preset").value = name;
  } catch (err) {
    $("ctl-preset-note").textContent = err.message;
  }
});

$("ctl-delete").addEventListener("click", async () => {
  const name = $("ctl-preset").value;
  const res = await fetch(`/api/control/presets/${encodeURIComponent(name)}`, { method: "DELETE" });
  $("ctl-preset-note").textContent = res.ok ? `deleted ${name}` : (await res.json()).detail;
  loadControlLists();
});

function drawControlHand(c, running) {
  const ctx = $("ctl-hand").getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  const angles = running && c.angles ? c.angles : config.gestures.rest.angles;
  drawSolidHand(ctx, anglesByName(angles), { view: ctlView, mirror: mirrored(), color: running ? TRACKED_COLOR : TARGET_COLOR });
}

attachRotate($("ctl-hand"), ctlView, () => state?.tools && drawControlHand(state.tools.control, state.tools.predictor.running && state.tools.predictor.control));

function drawTrace() {
  const ctx = $("ctl-trace").getContext("2d");
  const { width: w, height: h } = ctx.canvas;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#16181c";
  ctx.fillRect(0, 0, w, h);
  const xOf = (i) => (w * i) / (TRACE_STEPS - 1);
  const yOf = (v) => h - 6 - (h - 12) * Math.min(Math.max(v, 0), 1);
  ctx.lineWidth = 2;
  ctl.trace.forEach((s, i) => {
    if (!s.start && !s.release) return;
    ctx.strokeStyle = s.start ? "#45c27a" : "#e5484d";
    ctx.beginPath(); ctx.moveTo(xOf(i), 0); ctx.lineTo(xOf(i), h); ctx.stroke();
  });
  if (ctl.params) {
    ctx.setLineDash([6, 4]);
    for (const [key, color] of [["start_level", "#e0a02a"], ["hold_level", "#8b919c"], ["release_level", "#e5484d"]]) {
      if (key === "hold_level" && ctl.params.closure_mode !== "speed") continue;
      ctx.strokeStyle = color;
      ctx.beginPath(); ctx.moveTo(0, yOf(ctl.params[key])); ctx.lineTo(w, yOf(ctl.params[key])); ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  for (const [key, color] of [["effort", TRACKED_COLOR], ["closure", "#e6e6e6"]]) {
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctl.trace.forEach((s, i) => (i ? ctx.lineTo(xOf(i), yOf(s[key])) : ctx.moveTo(xOf(i), yOf(s[key]))));
    ctx.stroke();
  }
}

function drawControl(c, p) {
  if (!c) return;
  if (!ctl.params) {
    ctl.params = { ...c.params };
    buildControlParams();
    showControlParams();
  }
  const running = !!(p.running && p.control);
  $("ctl-stop").disabled = !running;
  $("ctl-reset").disabled = !running;
  $("ctl-status").textContent = running ? `Running ${p.model}, effort levels from ${c.levels_from}`
    : p.running ? `${p.model} is running without the control layer (started from another tab)` : "Stopped";
  if (p.error && running) $("ctl-error").textContent = p.error;
  if (running && c.step != null && c.step !== ctl.lastStep) {
    // A restart begins the step count again
    if (ctl.lastStep == null || c.step < ctl.lastStep) ctl.trace = [];
    else ctl.trace.push({ effort: c.effort, closure: c.closure, start: c.starts > ctl.starts, release: c.releases > ctl.releases });
    ctl.trace = ctl.trace.slice(-TRACE_STEPS);
    ctl.lastStep = c.step;
  }
  if (!running) ctl.lastStep = null;
  ctl.starts = c.starts ?? 0;
  ctl.releases = c.releases ?? 0;
  const name = (g) => g.replace("_", " ");
  $("ctl-label").innerHTML = !running ? "<small>not running</small>"
    : c.state === "active" ? `${name(c.grip)} ${Math.round(100 * c.closure)} %<small>locked, closure</small>`
    : "relaxed<small>waiting for a grip</small>";
  $("ctl-readout").innerHTML = running ? `<div>${Math.round(100 * c.effort)} %<small>effort</small></div>
    <div>${Math.round(100 * c.closure)} %<small>closure</small></div><div>${c.starts}<small>starts</small></div>
    <div>${c.releases}<small>releases</small></div><div>${c.opens}<small>open hand picked</small></div>
    <div>${c.distance == null ? "n/a" : c.distance.toFixed(1)}<small>distance${c.distance != null && ctl.params.max_distance > 0
      && c.distance > ctl.params.max_distance ? ", counted as rest" : ""}</small></div>` : "";
  if (currentTab !== "control") return;
  drawControlHand(c, running);
  drawTrace();
  drawProbs({ ...p, running, label: c.grip, threshold: ctl.params.threshold }, "ctl-probs");
}
