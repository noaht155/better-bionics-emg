// Control tab: the gesture model with the control layer (control.py) on its output. The server steps the
// controller inside the model's loop and sends its state with the rest; the page draws it and posts setting changes.
// Settings are split in basic and advanced (only the ones the mode uses), the mapping (rules mode's rule list) has
// its own page, and profiles save both.

const ALL_MODES = ["lock", "blend", "trigger", "rules"];
const CONTROL_PARAMS = [
  { key: "mode", basic: true, label: "Mode", options: ["rules", "trigger", "lock", "blend"], help: "" },
  { key: "pulse_level", basic: true, modes: ["trigger", "rules"], label: "Clench level", min: 0.02, max: 0.8, step: 0.01, help: "effort that counts as a clench" },
  { key: "gap_s", modes: ["trigger", "rules"], label: "Pause s", min: 0.2, max: 2, step: 0.05, help: "pause after the last short clench or flick that ends the count" },
  { key: "long_s", modes: ["trigger"], label: "Long clench s", min: 0.2, max: 2, step: 0.05, help: "a clench held this long is fist, shorter ones are counted" },
  { key: "release_with", modes: ["trigger"], label: "Release with", options: ["point", "open"], help: "two flicks of this gesture go back to picking a grip. point keeps the open hand for opening only" },
  { key: "pinch_clenches", modes: ["trigger"], label: "Pinch clenches", min: 1, max: 2, step: 1, help: "clenches for pinch, tripod and key are this, +1 and +2. 2 means a single clench picks nothing, most false picks were single clenches" },
  { key: "double_s", modes: ["trigger"], label: "Double flick s", min: 0.3, max: 2, step: 0.05, help: "two flicks within this time go back to picking a grip" },
  { key: "still_gate", modes: ["trigger", "rules"], label: "Arm still only", help: "clenches and classes only pick while the arm is still (accelerometer, limit from the model's session). Cut false picks during arm activity by about 2.5x in the replay" },
  { key: "smooth_s", modes: ["blend"], label: "Smoothing s", min: 0, max: 1.5, step: 0.05, help: "how long the grip probabilities are averaged over, longer is steadier but slower to change shape" },
  { key: "lock_s", modes: ["blend"], label: "Lock after s", min: 0, max: 2, step: 0.05, help: "one grip above the threshold this long locks the shape, 0 never locks" },
  { key: "threshold", label: "Threshold", min: 0, max: 0.99, step: 0.01, help: "probability at which a class counts (a vote, or a rule's class being on)" },
  { key: "vote_n", modes: ["lock", "trigger"], label: "Votes", min: 1, max: 15, step: 1, help: "steps in a row (50 ms each) the same grip has to win to start" },
  { key: "start_level", modes: ["lock", "blend"], label: "Start level", min: 0.01, max: 0.6, step: 0.01, help: "effort that counts as a rise from rest" },
  { key: "onset_s", modes: ["lock"], label: "Onset s", min: 0.1, max: 5, step: 0.05, help: "time after crossing the start level in which a grip can still be picked" },
  { key: "closure_mode", modes: ["lock"], label: "Closure", options: ["position", "speed"], help: "position: effort sets how far the hand is closed. speed: effort above the hold level closes, below opens" },
  { key: "gain", label: "Gain", min: 0, max: 10, step: 0.1, help: "closure per unit of effort (position) or per second (speed, trigger, rules)" },
  { key: "hold_level", modes: ["lock", "trigger", "rules"], label: "Hold level", min: 0, max: 0.6, step: 0.01, help: "effort above this closes, below it the hand holds (speed closure, trigger, rules)" },
  { key: "hold_on_relax", modes: ["lock", "blend"], label: "Hold on relax", help: "relaxing keeps the hand where it is instead of opening it" },
  { key: "release_mode", modes: ["lock", "blend"], label: "Release", options: ["rest_or_class", "either", "open_class", "rest"], help: "rest_or_class: rest or the open hand wins, or effort stays low. open_class: the open hand gesture. rest: effort stays low. either: open hand or low effort" },
  { key: "release_n", modes: ["lock", "blend"], label: "Release votes", min: 1, max: 10, step: 1, help: "steps in a row the open hand (or rest, in rest_or_class) has to win to release" },
  { key: "release_level", modes: ["lock", "blend"], label: "Release level", min: 0, max: 0.5, step: 0.01, help: "effort below which the hand counts as at rest" },
  { key: "release_s", modes: ["lock", "blend"], label: "Release s", min: 0, max: 3, step: 0.05, help: "time below the release level before the grip lets go" },
  { key: "max_distance", lda: true, label: "Max distance", min: 0, max: 15, step: 0.5, help: "LDA models only: windows farther than this from the grip they won count as rest, lower is stricter. 6 tested, 0 turns it off" },
  { key: "spring_w", label: "Spring", min: 0, max: 30, step: 0.5, help: "how fast the hand follows its target, 0 jumps straight there" },
];
// The mode's help line only describes the chosen mode
const MODE_HELP = {
  rules: "the mapping picks the pose; then squeezing closes, open opens, relaxing holds",
  trigger: "the fixed version of rules: long clench fist, 1 short pinch, 2 tripod, 3 key, point from the model, two point flicks go back",
  lock: "votes pick a grip while effort rises from rest, then it locks until a release",
  blend: "the hand closes with effort straight away towards a mix of the likely grips, the shape settles as the model firms up",
};
const BASIC_PARAMS = CONTROL_PARAMS.filter((d) => d.basic);
const ADVANCED_PARAMS = CONTROL_PARAMS.filter((d) => !d.basic);
const TRACE_STEPS = 200;
const ctlView = { yaw: 25, pitch: -20 };
const ctl = { params: null, rules: null, mappingName: "", models: [], presets: {}, mappings: {}, trace: [], lastStep: null, starts: 0, releases: 0 };

function showControlParams() {
  showParams(BASIC_PARAMS, "ctl-", ctl.params);
  showParams(ADVANCED_PARAMS, "ctl-", ctl.params);
  showParamsForMode();
}

// Only the settings the mode uses, and max distance only for LDA models (networks have no distance)
function showParamsForMode() {
  const network = $("ctl-model").value.endsWith(".pt");
  for (const d of CONTROL_PARAMS) {
    const visible = (d.modes || ALL_MODES).includes(ctl.params.mode) && !(d.lda && network);
    $(`ctl-${d.key}`).closest(".param").hidden = !visible;
  }
  $("ctl-mode").closest(".param").querySelector("small").textContent = MODE_HELP[ctl.params.mode] || "";
}

function sendControlParams() {
  showParamsForMode();
  postShowing("/api/control/params", ctl.params, "ctl-error");
}

function selectedModel() {
  return ctl.models.find((m) => m.name === $("ctl-model").value);
}

const filterValue = (name) => document.querySelector(`input[name=${name}]:checked`).value;

// Saved profiles or mappings, all of them or only those built on the chosen model
function fillSaved(select, saved, filter, fixed) {
  const model = $("ctl-model").value;
  const names = Object.keys(saved).filter((n) => n === fixed || filterValue(filter) === "all" || saved[n].model === model);
  const current = select.value;
  select.innerHTML = names.map((n) => `<option>${n}</option>`).join("") || "<option value=''>none for this model</option>";
  if (names.includes(current)) select.value = current;
}

async function loadControlLists() {
  const [lda, networks, presets, mappings] = await Promise.all(["/api/models", "/api/networks", "/api/control/presets", "/api/control/mappings"]
    .map((u) => fetch(u).then((x) => x.json())));
  // Calibrated networks take their effort levels from the session they were calibrated on. Sign networks also run
  // uncalibrated, with the levels of their newest training session
  ctl.models = [...lda, ...networks.filter((n) => n.calibrated_on || n.kind === "signs")];
  fillSelect($("ctl-model"), ctl.models, (m) => (m.calibrated_on
    ? `${m.name}, ${m.signs ? "sign " : ""}network calibrated on ${shortSession(m.calibrated_on)}`
    : m.kind === "signs" ? `${m.name}, sign network, not calibrated` : ldaLabel(m)));
  const running = state?.tools?.predictor?.running && state.tools.predictor.control && state.tools.predictor.model;
  if (running && ctl.models.some((m) => m.name === running)) $("ctl-model").value = running;
  ctl.presets = presets;
  ctl.mappings = mappings;
  fillSaved($("ctl-preset"), presets, "ctl-profile-filter", "default");
  fillSaved($("ctl-mapping-select"), mappings, "ctl-mapping-filter");
  if (ctl.params) showParamsForMode();
  drawRules();
  followModelDefault();
}

// A mapping nobody has touched (the default, never edited, loaded or saved) follows the chosen model's default, so a
// sign model doesn't start with rules on classes it doesn't have
let defaultModel = null;
async function followModelDefault() {
  const model = $("ctl-model").value;
  if (!ctl.rules || !model || model === defaultModel) return;
  if (ctl.mappingName && ctl.mappingName !== "default") return;
  const generic = await post("/api/control/default_rules", config.gesture_set);
  const wasDefault = ctl.mappingName === "default" || JSON.stringify(ctl.rules) === JSON.stringify(generic);
  if (!wasDefault) return;
  defaultModel = model;
  const rules = await post("/api/control/default_rules", selectedModel()?.classes || config.gesture_set);
  await setRules(rules, "default");
}

$("ctl-model").addEventListener("change", loadControlLists);
document.querySelectorAll("input[name=ctl-profile-filter]").forEach((r) => r.addEventListener("change", () => fillSaved($("ctl-preset"), ctl.presets, "ctl-profile-filter", "default")));
document.querySelectorAll("input[name=ctl-mapping-filter]").forEach((r) => r.addEventListener("change", () => fillSaved($("ctl-mapping-select"), ctl.mappings, "ctl-mapping-filter")));

$("ctl-start").addEventListener("click", async () => {
  $("ctl-haptics").textContent = "";
  const r = await postShowing("/api/control/start", { name: $("ctl-model").value, ...linkBody() }, "ctl-error");
  if (r) $("ctl-haptics").textContent = r.haptics || "";
});
$("ctl-stop").addEventListener("click", () => post("/api/model/stop").catch(() => {}));
$("ctl-reset").addEventListener("click", () => post("/api/control/reset").catch(() => {}));

// Profiles

$("ctl-load").addEventListener("click", async () => {
  const name = $("ctl-preset").value;
  const preset = ctl.presets?.[name];
  if (!preset || !ctl.params) return;
  ctl.params = { ...ctl.params, ...preset.params };
  showControlParams();
  sendControlParams();
  if (preset.rules) await setRules(preset.rules, `from profile ${name}`);
  $("ctl-preset-note").textContent = `loaded ${name}${preset.rules ? " and its mapping" : ""}`;
});

$("ctl-save").addEventListener("click", async () => {
  const name = $("ctl-preset-name").value.trim();
  try {
    await post("/api/control/presets", { name, params: ctl.params, rules: ctl.rules, model: $("ctl-model").value || null });
    $("ctl-preset-note").textContent = `saved ${name} with the mapping in use`;
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

// Mapping page

function showMappingPage(on) {
  closePicker();
  $("ctl-main").hidden = on;
  $("ctl-mapping-page").hidden = !on;
  if (on) drawRules();
}

$("ctl-mapping-open").addEventListener("click", () => showMappingPage(true));
$("ctl-mapping-back").addEventListener("click", () => showMappingPage(false));

async function setRules(rules, name) {
  ctl.rules = rules.map((r) => ({ ...r }));
  ctl.mappingName = name;
  drawRules();
  return sendRules();
}

async function sendRules() {
  $("ctl-mapping-error").textContent = "";
  try {
    await post("/api/control/rules", ctl.rules);
    return true;
  } catch (err) {
    $("ctl-mapping-error").textContent = err.message;
    return false;
  }
}

function changedRules() {
  if (ctl.mappingName && !ctl.mappingName.endsWith("(changed)")) ctl.mappingName += " (changed)";
  showMappingName();
  sendRules();
}

function showMappingName() {
  const n = ctl.rules?.length ?? 0;
  $("ctl-mapping-name").textContent = `${ctl.mappingName || "unsaved"}, ${n} rule${n === 1 ? "" : "s"}`;
}

const nice = (s) => s.replaceAll("_", " ");

// Triggers: clench and the chosen model's classes; a trigger the model doesn't have stays listed, marked
function triggerOptions(current) {
  const classes = (selectedModel()?.classes || []).filter((c) => c !== "rest");
  const options = ["clench", ...classes];
  if (current && !options.includes(current)) options.push(current);
  return options.map((t) => `<option value="${t}" ${t === current ? "selected" : ""}>${nice(t)}${t === "clench" || classes.includes(t) ? "" : " (not in this model)"}</option>`).join("");
}

// Then: the actions, the grips, then the protocol C library by category. A list of our own instead of a dropdown,
// since a dropdown can't tell which item the mouse is over and the hovered pose is drawn beside the list
const ACTIONS = { close: "close grip (while it stays on)", open: "open grip (while it stays on)", release: "release (back to choosing)" };

function actionGroups() {
  const groups = [["Actions", Object.entries(ACTIONS)],
    ["Grips", config.gesture_set.filter((g) => g !== "rest").map((g) => [g, nice(g)])]];
  for (const [cat, title] of Object.entries(config.categories)) {
    const items = Object.entries(config.library).filter(([, m]) => m.category === cat).map(([k, m]) => [k, m.name]);
    if (items.length) groups.push([title, items]);
  }
  return groups;
}

function actionLabel(action) {
  return ACTIONS[action] || config.library[action]?.name || nice(action);
}

// Grips take the drawn pose of the grip set, the library its own
function poseAngles(action) {
  if (ACTIONS[action]) return null;
  return config.gesture_set.includes(action) ? config.gestures[action].angles : config.library[action]?.angles;
}

function drawPosePreview(action) {
  const ctx = $("ctl-pose-preview").getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  const angles = poseAngles(action);
  $("ctl-pose-name").textContent = action ? actionLabel(action) : "";
  if (angles) drawSolidHand(ctx, anglesByName(angles), { view: ctlView, mirror: mirrored(), color: TRACKED_COLOR });
}

let pickerRow = null;
function openPicker(button) {
  pickerRow = Number(button.closest("tr").dataset.i);
  const current = ctl.rules[pickerRow].action;
  $("ctl-pose-list").innerHTML = actionGroups().map(([title, items]) => `<div class="pick-group">${title}</div>`
    + items.map(([k, label]) => `<div class="pick-item${k === current ? " current" : ""}" data-action="${k}">${label}</div>`).join("")).join("");
  const box = $("ctl-pose-picker");
  const r = button.getBoundingClientRect();
  box.hidden = false;
  box.style.left = `${Math.min(r.left, window.innerWidth - box.offsetWidth - 8)}px`;
  box.style.top = `${Math.max(8, Math.min(r.bottom + 4, window.innerHeight - box.offsetHeight - 8))}px`;
  drawPosePreview(current);
  $("ctl-pose-list").querySelector(".current")?.scrollIntoView({ block: "center" });
}

function closePicker() {
  $("ctl-pose-picker").hidden = true;
  pickerRow = null;
}

$("ctl-pose-list").addEventListener("mouseover", (e) => {
  const item = e.target.closest(".pick-item");
  if (item) drawPosePreview(item.dataset.action);
});
$("ctl-pose-list").addEventListener("mouseleave", () => pickerRow != null && drawPosePreview(ctl.rules[pickerRow].action));
$("ctl-pose-list").addEventListener("click", (e) => {
  const item = e.target.closest(".pick-item");
  if (!item || pickerRow == null) return;
  ctl.rules[pickerRow].action = item.dataset.action;
  closePicker();
  drawRules();
  changedRules();
});
document.addEventListener("mousedown", (e) => {
  if (!$("ctl-pose-picker").hidden && !e.target.closest("#ctl-pose-picker, .pose-pick")) closePicker();
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closePicker(); });

function drawRules() {
  if (!ctl.rules) return;
  showMappingName();
  if ($("ctl-mapping-page").hidden) return;
  $("ctl-rules").innerHTML = "<tr><th>If</th><th>How</th><th>Seconds or times</th><th>Then</th><th></th></tr>"
    + ctl.rules.map((r, i) => `<tr data-i="${i}">
      <td><select data-k="trigger">${triggerOptions(r.trigger)}</select></td>
      <td><select data-k="how"><option ${r.how === "held" ? "selected" : ""}>held</option><option ${r.how === "repeated" ? "selected" : ""}>repeated</option></select></td>
      <td><input data-k="value" type="number" min="${r.how === "held" ? 0 : 1}" max="${r.how === "held" ? 5 : 6}" step="${r.how === "held" ? 0.05 : 1}" value="${r.value}"></td>
      <td><button class="pose-pick">${actionLabel(r.action)} &#9662;</button></td>
      <td><button data-remove="${i}">Remove</button></td></tr>`).join("");
}

$("ctl-rules").addEventListener("change", (e) => {
  const row = e.target.closest("tr");
  const key = e.target.dataset.k;
  if (!row || !key) return;
  const rule = ctl.rules[Number(row.dataset.i)];
  rule[key] = key === "value" ? Number(e.target.value) : e.target.value;
  // A count is a whole number of times, a time can be fractional
  if (key === "how") rule.value = rule.how === "held" ? 0.5 : 2;
  drawRules();
  changedRules();
});

$("ctl-rules").addEventListener("click", (e) => {
  const pick = e.target.closest(".pose-pick");
  if (pick) {
    if (pickerRow === Number(pick.closest("tr").dataset.i)) closePicker();
    else openPicker(pick);
    return;
  }
  if (e.target.dataset.remove == null) return;
  ctl.rules.splice(Number(e.target.dataset.remove), 1);
  drawRules();
  changedRules();
});

$("ctl-rule-add").addEventListener("click", () => {
  ctl.rules.push({ trigger: "clench", how: "repeated", value: 4, action: "release" });
  drawRules();
  changedRules();
});

$("ctl-mapping-load").addEventListener("click", async () => {
  const name = $("ctl-mapping-select").value;
  const m = ctl.mappings[name];
  if (!m) return;
  if (await setRules(m.rules, name)) $("ctl-mapping-note").textContent = `loaded ${name}`;
});

$("ctl-mapping-save").addEventListener("click", async () => {
  const name = $("ctl-mapping-save-name").value.trim();
  try {
    await post("/api/control/mappings", { name, rules: ctl.rules, model: $("ctl-model").value || null });
    ctl.mappingName = name;
    showMappingName();
    $("ctl-mapping-note").textContent = `saved ${name}`;
    await loadControlLists();
    $("ctl-mapping-select").value = name;
  } catch (err) {
    $("ctl-mapping-note").textContent = err.message;
  }
});

$("ctl-mapping-delete").addEventListener("click", async () => {
  const name = $("ctl-mapping-select").value;
  const res = await fetch(`/api/control/mappings/${encodeURIComponent(name)}`, { method: "DELETE" });
  $("ctl-mapping-note").textContent = res.ok ? `deleted ${name}` : (await res.json()).detail;
  loadControlLists();
});

$("ctl-mapping-default").addEventListener("click", async () => {
  const rules = await post("/api/control/default_rules", selectedModel()?.classes || config.gesture_set);
  if (await setRules(rules, "default")) $("ctl-mapping-note").textContent = "default mapping for this model";
});

// Drawing

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
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, w, h);
  const xOf = (i) => (w * i) / (TRACE_STEPS - 1);
  const yOf = (v) => h - 6 - (h - 12) * Math.min(Math.max(v, 0), 1);
  ctx.lineWidth = 2;
  ctl.trace.forEach((s, i) => {
    if (!s.start && !s.release) return;
    ctx.strokeStyle = s.start ? COLORS.good : COLORS.bad;
    ctx.beginPath(); ctx.moveTo(xOf(i), 0); ctx.lineTo(xOf(i), h); ctx.stroke();
  });
  if (ctl.params) {
    ctx.setLineDash([6, 4]);
    const clenches = ctl.params.mode === "trigger" || ctl.params.mode === "rules";
    const lines = clenches ? [["pulse_level", COLORS.warn], ["hold_level", COLORS.muted]]
      : [["start_level", COLORS.warn], ["hold_level", COLORS.muted], ["release_level", COLORS.bad]];
    for (const [key, color] of lines) {
      if (key === "hold_level" && ctl.params.closure_mode !== "speed" && !clenches) continue;
      ctx.strokeStyle = color;
      ctx.beginPath(); ctx.moveTo(0, yOf(ctl.params[key])); ctx.lineTo(w, yOf(ctl.params[key])); ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  for (const [key, color] of [["effort", TRACKED_COLOR], ["closure", COLORS.accent]]) {
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
    buildParams($("ctl-basic"), BASIC_PARAMS, "ctl-", () => ctl.params, sendControlParams);
    buildParams($("ctl-params"), ADVANCED_PARAMS, "ctl-", () => ctl.params, sendControlParams);
    showControlParams();
  }
  if (!ctl.rules && c.rules) {
    ctl.rules = c.rules.map((r) => ({ ...r }));
    drawRules();
    followModelDefault();
  }
  const running = !!(p.running && p.control);
  $("ctl-output").hidden = !running;
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
  const counting = ctl.params.mode === "trigger" || ctl.params.mode === "rules";
  $("ctl-label").innerHTML = !running ? "<small>not running</small>"
    : c.state === "active" ? `${nice(c.grip)} ${Math.round(100 * c.closure)} %<small>locked, closure</small>`
    : c.state === "blend" ? `${c.grip ? nice(c.grip) : "relaxed"} ${Math.round(100 * c.closure)} %<small>blending, closest grip and closure</small>`
    : counting ? `relaxed<small>${ctl.params.still_gate && c.moving ? "arm moving, clenches ignored" : c.pulses ? `${c.pulses} clench${c.pulses > 1 ? "es" : ""} so far` : "waiting for a pattern"}</small>`
    : "relaxed<small>waiting for a grip</small>";
  flashOnChange($("ctl-label"), running ? `${c.state} ${c.grip}` : "");
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
