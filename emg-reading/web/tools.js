// Tabs for the existing tools: live signals (live_plot.py), calibration (calibrate.py), the gesture model
// (ml/gesture_model.py), the general network (ml/train_ringnet.py, ml/networks.py), haptics and the output test (haptics.py, esp32_link.py) and the connection check
// (check_connection.py).

const LINK_KEY = "esp32-link";
const SPECTRUM_MS = 250;

// Tabs

// The setup and troubleshooting pages sit under System. Cursor is archived: only listed with Show archived on
const SYSTEM_TABS = ["signals", "connection", "haptics", "calibrate", "cursor"];
const SYSTEM_KEY = "system-tab";
const ARCHIVED_KEY = "show-archived";

function stored(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}

function store(key, value) {
  try { localStorage.setItem(key, value); } catch { /* private window, nothing is remembered */ }
}

const showArchived = () => $("show-archived").checked;

function showTab(name) {
  if (name === "system") {
    const last = stored(SYSTEM_KEY);
    name = SYSTEM_TABS.includes(last) && (last !== "cursor" || showArchived()) ? last : "signals";
  }
  currentTab = name;
  const system = SYSTEM_TABS.includes(name);
  if (system) store(SYSTEM_KEY, name);
  $("subtabs").hidden = !system;
  document.querySelectorAll("#tabs button").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === name || (system && b.dataset.tab === "system"));
  });
  document.querySelectorAll("#subtabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => { t.hidden = t.id !== `tab-${name}`; });
  // The camera preview and the signal stream only run while their tab is open
  if (name === "record") showView();
  else $("video").src = "";
  if (name === "signals") openSignals();
  else closeSignals();
  if (name === "train") loadTrainLists();
  if (name === "control") loadControlLists();
  if (name === "cursor") loadCursorSessions();
  moveTabInk();
}

document.querySelectorAll("#tabs button, #subtabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

function applyArchived() {
  document.querySelectorAll("#subtabs .archived").forEach((b) => { b.hidden = !showArchived(); });
  if (!showArchived() && currentTab === "cursor") showTab("signals");
}

$("show-archived").checked = stored(ARCHIVED_KEY) === "1";
$("show-archived").addEventListener("change", () => {
  store(ARCHIVED_KEY, showArchived() ? "1" : "0");
  applyArchived();
});
applyArchived();

function moveTabInk() {
  const b = document.querySelector("#tabs button.active");
  b.scrollIntoView({ block: "nearest", inline: "nearest" });
  const ink = $("tab-ink");
  ink.style.left = `${b.offsetLeft}px`;
  ink.style.top = `${b.offsetTop + b.offsetHeight - 2}px`;
  ink.style.width = `${b.offsetWidth}px`;
}

// Tab widths change once the web font has loaded and when the bar wraps
moveTabInk();
document.fonts.ready.then(moveTabInk);
window.addEventListener("resize", moveTabInk);

// The bar only shows while its job runs, an empty track looked like something stuck at 0
function progress(bar, tool) {
  $(bar).parentElement.hidden = !tool.running;
  $(bar).style.width = tool.running && tool.total && tool.left != null ? `${100 * (1 - tool.left / tool.total)}%` : "0";
}

// Live signals, drawn like live_plot.py: a 4 s sweep per channel with a shared y scale

const sig = { ws: null, raw: null, filtered: null, pos: 0, spectrum: null, timer: null };

function resetSweep() {
  const n = 4 * config.rate;
  sig.raw = config.channels.map(() => new Float64Array(n).fill(NaN));
  sig.filtered = config.channels.map(() => new Float64Array(n).fill(NaN));
  sig.pos = 0;
}

function openSignals() {
  if (sig.ws) return;
  resetSweep();
  sig.ws = new WebSocket(`ws://${location.host}/ws/signals`);
  sig.ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    const n = sig.raw[0].length;
    const k = Math.min(msg.raw[0].length, n);
    const gap = Math.floor(config.rate / 20);
    for (let c = 0; c < msg.raw.length; c++) {
      const len = msg.raw[c].length;
      for (let i = 0; i < k; i++) {
        const at = (sig.pos + i) % n;
        sig.raw[c][at] = msg.raw[c][len - k + i];
        sig.filtered[c][at] = msg.filtered[c][len - k + i];
      }
      for (let i = 0; i < gap; i++) {
        sig.raw[c][(sig.pos + k + i) % n] = NaN;
        sig.filtered[c][(sig.pos + k + i) % n] = NaN;
      }
    }
    sig.pos = (sig.pos + k) % n;
    if (!$("sig-spectrum").checked) drawSweep();
  };
  sig.timer = setInterval(pollSpectrum, SPECTRUM_MS);
}

function closeSignals() {
  if (sig.ws) sig.ws.close();
  sig.ws = null;
  clearInterval(sig.timer);
}

async function pollSpectrum() {
  if (!$("sig-spectrum").checked) return;
  const res = await fetch(`/api/spectrum?filtered=${$("sig-filtered").checked}`);
  sig.spectrum = await res.json();
  drawSpectrum();
}

function channelRows(ctx) {
  // Bitmap matches the canvas's size on screen, otherwise a wide or high-dpi screen stretches it and blurs the traces
  const canvas = ctx.canvas;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const rowH = h / config.channels.length;
  ctx.font = `11px ${MONO}`;
  return { w, rowH };
}

function drawSweep() {
  const ctx = $("signals").getContext("2d");
  const { w, rowH } = channelRows(ctx);
  const filtered = $("sig-filtered").checked;
  const data = filtered ? sig.filtered : sig.raw;
  const n = data[0].length;
  // Raw channels sit on a large DC offset, so each is centred on its own mean
  const centred = data.map((x) => {
    let sum = 0, count = 0;
    for (const v of x) if (!Number.isNaN(v)) { sum += v; count++; }
    const mean = count ? sum / count : 0;
    return x.map((v) => v - mean);
  });
  let lim = 1;
  for (const x of centred) for (const v of x) if (Math.abs(v) > lim) lim = Math.abs(v);
  const left = 90;
  centred.forEach((x, c) => {
    const mid = rowH * c + rowH / 2;
    let sq = 0, count = 0;
    for (const v of x) if (!Number.isNaN(v)) { sq += v * v; count++; }
    ctx.fillStyle = COLORS.text;
    ctx.fillText(`ch ${config.channels[c]}`, 8, mid - 4);
    ctx.fillStyle = COLORS.muted;
    ctx.fillText(`rms ${count ? Math.sqrt(sq / count).toFixed(1) : "-"} µV`, 8, mid + 13);
    ctx.strokeStyle = COLORS.line;
    ctx.beginPath(); ctx.moveTo(left, mid); ctx.lineTo(w, mid); ctx.stroke();
    ctx.strokeStyle = TRACKED_COLOR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    let drawing = false;
    for (let i = 0; i < n; i++) {
      if (Number.isNaN(x[i])) { drawing = false; continue; }
      const px = left + ((w - left) * i) / n;
      const py = mid - (x[i] / lim) * (rowH / 2 - 4);
      if (drawing) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      drawing = true;
    }
    ctx.stroke();
  });
  ctx.fillStyle = COLORS.muted;
  ctx.fillText(`±${lim.toFixed(0)} µV`, w - 90, 14);
}

function drawSpectrum() {
  const s = sig.spectrum;
  if (!s || !s.f.length) return;
  const ctx = $("signals").getContext("2d");
  const { w, rowH } = channelRows(ctx);
  const left = 90;
  const fMax = s.f[s.f.length - 1];
  let lo = Infinity, hi = 0;
  for (const m of s.mag) for (const v of m.slice(1)) { if (v > 0) lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const logLo = Math.log10(lo), logHi = Math.log10(hi);
  const xOf = (f) => left + ((w - left) * f) / fMax;
  s.mag.forEach((m, c) => {
    const top = rowH * c;
    ctx.fillStyle = COLORS.text;
    ctx.fillText(`ch ${config.channels[c]}`, 8, top + rowH / 2);
    // Mains and its harmonics, where the notches sit
    ctx.strokeStyle = COLORS.mains;
    for (const f of [60, 120, 180]) { ctx.beginPath(); ctx.moveTo(xOf(f), top); ctx.lineTo(xOf(f), top + rowH); ctx.stroke(); }
    ctx.strokeStyle = TRACKED_COLOR;
    ctx.beginPath();
    m.forEach((v, i) => {
      if (i === 0) return;
      const t = (Math.log10(Math.max(v, lo)) - logLo) / (logHi - logLo || 1);
      const py = top + rowH - 4 - t * (rowH - 8);
      if (i === 1) ctx.moveTo(xOf(s.f[i]), py); else ctx.lineTo(xOf(s.f[i]), py);
    });
    ctx.stroke();
  });
  ctx.fillStyle = COLORS.muted;
  for (let f = 0; f <= fMax; f += 50) ctx.fillText(`${f} Hz`, xOf(f) - 12, rowH * config.channels.length - 2);
}

function toggleSignal(id) {
  $(id).checked = !$(id).checked;
  $(id).dispatchEvent(new Event("change"));
}

$("sig-filtered").addEventListener("change", () => { if ($("sig-spectrum").checked) pollSpectrum(); else drawSweep(); });
$("sig-spectrum").addEventListener("change", () => { if ($("sig-spectrum").checked) pollSpectrum(); else drawSweep(); });

document.addEventListener("keydown", (e) => {
  if (currentTab !== "signals" || e.target.tagName === "INPUT") return;
  if (e.key.toLowerCase() === "f") toggleSignal("sig-filtered");
  if (e.key.toLowerCase() === "s") toggleSignal("sig-spectrum");
});

// Calibration

function showCurrentCalibration() {
  const cal = config.calibration;
  if (!cal) {
    $("cal-current").textContent = "none yet";
    return;
  }
  $("cal-current").textContent = ["ch    rest     max   ratio"].concat(Object.entries(cal).map(([ch, v]) =>
    `${ch}  ${v.rest.toFixed(1).padStart(6)}  ${v.max.toFixed(1).padStart(6)}  ${(v.max / v.rest).toFixed(1).padStart(5)}x`)).join("\n");
}

$("cal-start").addEventListener("click", () => {
  postShowing("/api/calibrate", { seconds: Number($("cal-seconds").value), car: $("cal-car").checked }, "cal-error");
});

let calWasRunning = false;
async function drawCalibration(t) {
  $("cal-start").disabled = t.running || !!state.session;
  const prompt = t.prompt ? `${t.prompt}${t.countdown ? `  ${t.countdown}...` : ""}` : "";
  $("cal-prompt").textContent = t.running ? prompt : "";
  progress("cal-bar", t);
  if (t.error) $("cal-error").textContent = t.error;
  $("cal-result").textContent = (t.lines || []).join("\n");
  if (calWasRunning && !t.running) {
    config = await (await fetch("/api/config")).json();
    showCurrentCalibration();
  }
  calWasRunning = t.running;
}

// Haptics and the output test

function linkBody() {
  const mode = $("esp-link").value;
  if (mode === "usb") return { port: $("esp-port").value.trim() || null };
  if (mode === "wifi") return { wifi: $("esp-ip").value.trim() || config.esp32_wifi_ip };
  return {};
}

function saveLink() {
  const link = { mode: $("esp-link").value, port: $("esp-port").value, ip: $("esp-ip").value };
  try { localStorage.setItem(LINK_KEY, JSON.stringify(link)); } catch { /* private window */ }
  $("esp-port-label").hidden = link.mode !== "usb";
  $("esp-ip-label").hidden = link.mode !== "wifi";
}

for (const id of ["esp-link", "esp-port", "esp-ip"]) $(id).addEventListener("input", saveLink);

$("hap-start").addEventListener("click", () => postShowing("/api/haptics/start", linkBody(), "hap-error"));
$("hap-stop").addEventListener("click", () => postShowing("/api/haptics/stop", undefined, "hap-error"));
$("esp-test").addEventListener("click", () => postShowing("/api/esp32/test", linkBody(), "hap-error"));

function drawOutputs(h) {
  const { ctx, w, h: hgt } = sharpen($("outputs"));
  ctx.clearRect(0, 0, w, hgt);
  const outputs = h.outputs || [];
  if (!outputs.length) {
    drawIdle(ctx, "start haptics or the output test to see the outputs");
    return;
  }
  const slot = w / outputs.length;
  const top = 10, bottom = 44;
  const plotH = hgt - top - bottom;
  ctx.font = `13px ${MONO}`;
  outputs.forEach((chs, i) => {
    const duty = h.duties?.[i] ?? 0;
    const x = i * slot + slot * 0.15;
    const bw = slot * 0.7;
    ctx.fillStyle = COLORS.raised;
    ctx.fillRect(x, top, bw, plotH);
    const lost = chs.some((ch) => h.lost?.[config.channels.indexOf(ch)]);
    ctx.fillStyle = lost ? COLORS.bad : TRACKED_COLOR;
    ctx.fillRect(x, top + plotH * (1 - duty / 100), bw, plotH * duty / 100);
    ctx.fillStyle = COLORS.text;
    ctx.fillText(`out ${i}  ${duty}%`, x, hgt - bottom + 18);
    ctx.fillStyle = COLORS.muted;
    ctx.fillText(`ch ${chs.join("+")}`, x, hgt - bottom + 35);
  });
}

function drawHaptics(h, test) {
  $("hap-start").disabled = h.running || test.running;
  $("hap-stop").disabled = !h.running;
  $("esp-test").disabled = h.running || test.running;
  if (h.running) {
    const source = h.with_model ? `${h.source}, started with the model` : h.source;
    $("hap-status").textContent = h.link
      ? `Running on ${h.link}, ${source}, common average ${h.car ? "on" : "off"}, watchdog fired ${h.watchdog} times`
      : "Connecting to the ESP32...";
  } else {
    $("hap-status").textContent = test.running ? "Output test running..." : "Stopped";
  }
  if (h.error || test.error) $("hap-error").textContent = h.error || test.error;
  drawOutputs(h);
  $("esp-log").textContent = (test.lines || []).join("\n");
}

// Gesture model: training and live test

// Session folders start with date and time, that's enough to tell them apart
const shortSession = (name) => name.slice(0, 17).replace("_", " ");

function ldaLabel(m) {
  const acc = m.balanced != null ? `, ${Math.round(100 * m.balanced)} % balanced on unseen sessions` : "";
  return `${m.name}, trained on ${m.sessions.map(shortSession).join(", ")}${acc}`;
}

// Grip sessions (protocols A to D) and sign sessions (E, F) train separate models, the server refuses a mix too
const signSession = (s) => s.protocol === "E" || s.protocol === "F";
const GRIP_PROTOCOLS = ["A", "B", "C", "D"];
const SIGN_PROTOCOLS = ["E", "F"];
let trainSessions = [];
// Sign networks start from the newest general network (networks are listed newest first)
let newestGeneral = null;

function networkLabel(n) {
  if (n.kind === "signs") return `${n.name}, sign network from ${n.sessions.length} sessions, not calibrated`;
  const score = n.balanced != null ? `, ${pct(n.balanced)} balanced on its other repetitions` : "";
  const seen = n.seen_in_training ? " (session was in training, optimistic)" : "";
  return `${n.name}, ${n.signs ? "sign " : ""}network calibrated on ${shortSession(n.calibrated_on)}${score}${seen}`;
}

async function loadTrainLists() {
  const [sessions, models, networks] = await Promise.all(["/api/sessions", "/api/models", "/api/networks"]
    .map((u) => fetch(u).then((x) => x.json())));
  const table = $("train-sessions");
  const checked = new Set([...table.querySelectorAll("input:checked")].map((c) => c.value));
  const listed = new Set([...table.querySelectorAll("input")].map((c) => c.value));
  const signsTicked = trainSessions.some((s) => checked.has(s.name) && signSession(s));
  trainSessions = sessions.filter((s) => s.gestures > 0 || s.signs > 0);
  table.innerHTML = "<tr><th></th><th>Session</th><th>Arm</th><th>Postures</th><th>Placement</th><th>Length</th><th>Protocol</th></tr>";
  for (const s of trainSessions) {
    const tr = document.createElement("tr");
    // Keep the choice for sessions already listed. A grip session recorded since then starts ticked if it finished
    // and no sign sessions are ticked
    const on = listed.has(s.name) ? checked.has(s.name) : s.completed && !signSession(s) && !signsTicked;
    tr.innerHTML = `<td><input type="checkbox" value="${s.name}" ${on ? "checked" : ""}></td><td>${s.name}</td>
      <td>${s.band_arm ?? ""}${s.port_facing ? `, port to ${s.port_facing}` : ""}</td><td>${(s.postures || []).join(", ")}</td><td>${s.placement || ""}</td>
      <td>${s.seconds ? fmtTime(s.seconds) : ""}${s.completed ? "" : " (stopped early)"}</td>
      <td>${s.protocol}</td>`;
    table.append(tr);
  }
  if (!trainSessions.length) table.innerHTML = "<tr><td>No sessions with gestures yet, record one on the Record tab.</td></tr>";
  const present = new Set(trainSessions.map((s) => s.protocol));
  const picks = [["All grip sessions (A to D)", GRIP_PROTOCOLS], ...GRIP_PROTOCOLS.map((p) => [`All ${p}`, [p]]),
    ["All sign sessions (E, F)", SIGN_PROTOCOLS], ...SIGN_PROTOCOLS.map((p) => [`All ${p}`, [p]])]
    .filter(([, protocols]) => protocols.some((p) => present.has(p)));
  $("train-pick").innerHTML = picks.map(([label, protocols]) => `<button data-protocols="${protocols.join("")}">${label}</button>`).join("")
    + "<button data-protocols=''>Deselect all</button>";
  updateTrainForm();

  newestGeneral = networks.find((n) => n.kind === "general") || null;
  fillSelect($("net-base"), networks.filter((n) => n.kind === "general" || n.kind === "signs"),
    (n) => `${n.name}, ${n.kind === "signs" ? "signs from " : ""}${n.sessions.length} sessions, ${n.epochs} epochs`);
  fillSelect($("net-session"), trainSessions, (s) => `${s.name}, protocol ${s.protocol}${s.completed ? "" : " (stopped early)"}`);
  const live = [...models, ...networks.filter((n) => n.kind === "calibrated" || n.kind === "signs")];
  fillSelect($("model-select"), live, (m) => (m.name.endsWith(".pt") ? networkLabel(m) : `LDA ${ldaLabel(m)}`));
  // A model that was just trained is already running, show that one
  const running = state?.tools?.predictor?.running && state.tools.predictor.model;
  if (running && live.some((m) => m.name === running)) $("model-select").value = running;
}

function tickedSessions() {
  const names = new Set([...$("train-sessions").querySelectorAll("input:checked")].map((c) => c.value));
  return trainSessions.filter((s) => names.has(s.name));
}

// Completed sessions of the picked protocols. Picking the other kind unticks the ticked ones first
$("train-pick").addEventListener("click", (e) => {
  const protocols = e.target.dataset?.protocols;
  if (protocols == null) return;
  for (const c of $("train-sessions").querySelectorAll("input")) {
    const s = trainSessions.find((x) => x.name === c.value);
    c.checked = Boolean(protocols) && s.completed && protocols.includes(s.protocol);
  }
  $("train-pick-note").textContent = "";
  updateTrainForm();
});

// Ticking a session of the other kind unticks the rest
$("train-sessions").addEventListener("change", (e) => {
  if (!e.target.checked) return updateTrainForm();
  const sign = signSession(trainSessions.find((s) => s.name === e.target.value));
  let cleared = 0;
  for (const c of $("train-sessions").querySelectorAll("input:checked")) {
    if (signSession(trainSessions.find((s) => s.name === c.value)) !== sign) { c.checked = false; cleared++; }
  }
  $("train-pick-note").textContent = cleared
    ? `Unticked ${cleared} ${sign ? "grip" : "sign"} session${cleared > 1 ? "s" : ""}: grip and sign sessions train separate models.` : "";
  updateTrainForm();
});

let trainFamily = null;
function updateTrainForm() {
  const ticked = tickedSessions();
  const signs = ticked.some(signSession);
  const network = $("train-kind").value === "network";
  $("train-lda-options").hidden = network;
  $("train-net-options").hidden = !network;
  // Sign networks only fine-tune a general one, they settle in more epochs
  if (signs !== trainFamily) $("net-epochs").value = signs ? 30 : 15;
  trainFamily = signs;
  $("train-kind-note").textContent = !network ? "Trains, tests each session on a model trained on the others, saves it and makes it the live one."
    : signs ? (newestGeneral ? `Starts from the newest general network (${newestGeneral.name}, trained on ${newestGeneral.sessions.length} grip sessions), `
      + "which already knows what this band's EMG looks like, and fine-tunes it on the signs every ticked session has. Calibrate it below before live use."
      : "Needs a general network to start from: train one on grip sessions first.")
    : "Trains a new general network on the ticked sessions. Calibrate it below before live use.";
  // Only the positions every ticked session has can be trained, those are the ones to leave out
  const shared = ticked.length ? ticked.map((s) => new Set(s.positions || []))
    .reduce((a, b) => new Set([...a].filter((x) => b.has(x)))) : new Set();
  const box = $("train-exclude-list");
  const was = new Set([...box.querySelectorAll("input:checked")].map((c) => c.value));
  box.innerHTML = [...shared].sort().map((p) => `<label class="check"><input type="checkbox" value="${p}"
    ${was.has(p) ? "checked" : ""}> ${p.replaceAll("_", " ")}</label>`).join("")
    || `<small>${!ticked.length ? "tick sessions first"
      : ticked.some((s) => !s.positions) ? "restart the app, the running server is older than this page"
      : "the ticked sessions share no positions"}</small>`;
  box.hidden = !$("train-exclude").checked;
}

$("train-kind").addEventListener("change", updateTrainForm);
$("train-exclude").addEventListener("change", updateTrainForm);

$("train-start").addEventListener("click", () => {
  const ticked = tickedSessions();
  const sessions = ticked.map((s) => s.name);
  const exclude = $("train-exclude").checked
    ? [...$("train-exclude-list").querySelectorAll("input:checked")].map((c) => c.value) : [];
  $("train-error").textContent = "";
  $("net-error").textContent = "";
  if ($("train-kind").value === "lda") {
    postShowing("/api/train", {
      sessions, log: $("train-log").checked, accel: $("train-accel").checked, extended: $("train-extended").checked,
      vote: Number($("train-vote").value), exclude,
    }, "train-error");
    return;
  }
  const signs = ticked.some(signSession);
  const body = { job: signs ? "signs" : "train", epochs: Number($("net-epochs").value), sessions, exclude };
  if (signs) body.network = newestGeneral?.name;
  postShowing("/api/network/run", body, "net-error");
});

// Haptics start with the model, on the ESP32 link set on the Haptics tab
async function useModel(name, error, note) {
  $(note).textContent = "";
  const r = await postShowing("/api/model/use", { name, ...linkBody() }, error);
  if (r) $(note).textContent = r.haptics || "";
}

$("model-use").addEventListener("click", () => useModel($("model-select").value, "model-error", "model-haptics"));
$("model-stop").addEventListener("click", () => post("/api/model/stop").catch(() => {}));
$("model-threshold").addEventListener("change", () => {
  const value = Math.min(0.99, Math.max(0, Number($("model-threshold").value)));
  post("/api/model/threshold", { value }).catch((err) => { $("model-error").textContent = err.message; });
});

let trainWasRunning = false;
// Fills a log box, and keeps it scrolled to the newest line unless the user has scrolled up to read
function showLog(pre, lines) {
  const text = lines.join("\n");
  if (pre.textContent === text) return;
  const atEnd = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 20;
  pre.textContent = text;
  if (atEnd) pre.scrollTop = pre.scrollHeight;
}

function drawTrain(t) {
  $("train-step").textContent = t.running ? `${t.step || "starting"}...` : "";
  $("train-progress").hidden = !t.running;
  $("train-bar").style.width = `${100 * (t.share ?? 0)}%`;
  $("train-log-box").hidden = !(t.log || []).length;
  showLog($("train-log"), t.log || []);
  if (t.error) $("train-error").textContent = t.error;
  $("train-result").textContent = (t.lines || []).join("\n");
  if (trainWasRunning && !t.running && currentTab === "train") loadTrainLists();
  trainWasRunning = t.running;
}

function predictionLabel(p) {
  if (!p.running) return "";
  if (!p.label) return "<small>waiting for EMG</small>";
  const name = (g) => g.replace("_", " ");
  if (p.label === "unsure") {
    const [best, prob] = Object.entries(p.probs || {}).sort((a, b) => b[1] - a[1])[0] || ["-", 0];
    // The label is the vote over the last windows, so the newest window alone can be above the threshold
    const guess = `best guess ${name(best)} at ${Math.round(100 * prob)} %`;
    return `unsure<small>${guess}, not steadily above the ${p.threshold} threshold</small>`;
  }
  const conf = Math.round(100 * (p.probs?.[p.label] ?? 0));
  return `${name(p.label)}<small>${conf} % sure, this window says ${name(p.raw)}</small>`;
}

function drawProbs(p, canvas = "live-probs") {
  const { ctx, w, h } = sharpen($(canvas));
  ctx.clearRect(0, 0, w, h);
  if (!p.running || !p.probs) {
    drawIdle(ctx, "start a model to see its class probabilities");
    return;
  }
  const classes = p.classes;
  const rowH = h / classes.length;
  const left = 110;
  ctx.font = `13px ${SANS}`;
  classes.forEach((c, i) => {
    const v = p.probs[c] ?? 0;
    const y = i * rowH;
    ctx.fillStyle = c === p.label ? COLORS.text : COLORS.muted;
    ctx.fillText(c.replace("_", " "), 4, y + rowH * 0.65);
    ctx.fillStyle = COLORS.raised;
    ctx.fillRect(left, y + 3, w - left, rowH - 6);
    ctx.fillStyle = c === p.label ? COLORS.accent : COLORS.lineStrong;
    ctx.fillRect(left, y + 3, (w - left) * v, rowH - 6);
  });
  // A gesture's bar has to cross this line before it is shown
  const x = left + (w - left) * (p.threshold ?? 0);
  ctx.strokeStyle = COLORS.text;
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
}

// The virtual hand shows the predicted gesture's target pose
function drawModelView() {
  const p = state.tools?.predictor;
  const { ctx } = sharpen($("model-hand"), false);
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  if (!p?.running) {
    $("model-label").innerHTML = "<small>no model running, train or pick one on the Train tab</small>";
    return;
  }
  $("model-label").innerHTML = predictionLabel(p);
  flashOnChange($("model-label"), String(p.label));
  const g = p.label && config.gestures[p.label];
  if (g) drawSolidHand(ctx, anglesByName(g.angles), { view: solidView, mirror: mirrored(), color: TRACKED_COLOR });
}

function drawPredictor(p) {
  $("model-stop").disabled = !p.running;
  const box = $("model-threshold");
  if (p.threshold != null && document.activeElement !== box) box.value = p.threshold;
  $("model-status").textContent = p.running ? `Running ${p.model}` : "No model running";
  if (p.error) $("model-error").textContent = p.error;
  $("live-label").innerHTML = predictionLabel(p);
  flashOnChange($("live-label"), p.running ? String(p.label) : "");
  // No empty plot taking up the page while nothing runs
  $("live-probs").hidden = !p.running;
  drawProbs(p);
  // Rewriting an option redraws an open dropdown, so only touch it when the model starts or stops
  const option = $("model-option");
  if (option.disabled === !!p.running) {
    option.disabled = !p.running;
    option.textContent = p.running ? "Model prediction" : "Model prediction (no model running)";
  }
}

// General network: evaluation, training and calibration run as a background job, live test through the predictor

const pct = (v) => (v == null ? "-" : `${(100 * v).toFixed(1)} %`);

function fillSelect(select, items, label) {
  const current = select.value;
  select.innerHTML = items.map((m) => `<option value="${m.name}">${label(m)}</option>`).join("")
    || "<option value=''>none yet</option>";
  if (items.some((m) => m.name === current)) select.value = current;
}

// Settings from a list of {key, label, help} plus options (a dropdown), min, max and step (a slider) or neither (a
// checkbox). getParams returns the object the values live in, onChange runs after the user changes one
function buildParams(box, defs, prefix, getParams, onChange) {
  box.innerHTML = defs.map((d) => {
    let input;
    if (d.options) input = `<select id="${prefix}${d.key}">${d.options.map((o) => `<option>${o}</option>`).join("")}</select>`;
    else if (d.min == null) input = `<input type="checkbox" id="${prefix}${d.key}">`;
    else input = `<input type="range" id="${prefix}${d.key}" min="${d.min}" max="${d.max}" step="${d.step}">`;
    return `<label class="param"><span>${d.label}</span>${input}<output></output><small>${d.help}</small></label>`;
  }).join("");
  for (const d of defs) {
    $(`${prefix}${d.key}`).addEventListener("input", (e) => {
      const el = e.target;
      getParams()[d.key] = d.options ? el.value : d.min == null ? el.checked : Number(el.value);
      showParams(defs, prefix, getParams());
      onChange();
    });
  }
}

function showParams(defs, prefix, params) {
  for (const d of defs) {
    const el = $(`${prefix}${d.key}`);
    const v = params[d.key];
    if (d.min == null && !d.options) el.checked = v;
    else el.value = v;
    if (!d.options && d.min != null) el.nextElementSibling.textContent = v;
  }
}

// Calibration and evaluation run as a network job like training
function runNetwork(job) {
  const body = { job, epochs: Number($("net-epochs").value) };
  if (job === "calibrate") {
    body.network = $("net-base").value;
    body.session = $("net-session").value;
    body.reps = $("net-reps").value.split(/[\s,]+/).filter(Boolean).map(Number);
  }
  postShowing("/api/network/run", body, "net-error");
}

$("net-eval-newest").addEventListener("click", () => runNetwork("evaluate_newest"));
$("net-eval").addEventListener("click", () => runNetwork("evaluate"));
$("net-calibrate").addEventListener("click", () => runNetwork("calibrate"));
$("net-stop").addEventListener("click", () => post("/api/network/stop").catch(() => {}));

function networkTable(job, r) {
  if (job.startsWith("evaluate") && r.grips) {
    const rows = Object.entries(r.grips).map(([name, m]) => `<tr><td>${name}</td><td>${pct(m.balanced)}</td>
      <td>${pct(m.grips)}</td><td>${pct(m.rest_false)}</td><td>${pct(m.wrong)}</td><td>${pct(m.answers)}</td></tr>`);
    const angles = Object.entries(r.angles).map(([name, m]) => `<tr><td>finger angles, ${name}</td>
      <td colspan="5">${m.mae.toFixed(1)} deg mean error, r ${m.r.toFixed(2)}</td></tr>`);
    return `<tr><th>held out: ${r.held.map((n) => n.slice(0, 17)).join(", ")}</th><th>balanced</th><th>grips only</th>
      <th>rest taken for a grip</th><th>wrong grip shown</th><th>answers given</th></tr>${rows.join("")}${angles.join("")}`;
  }
  if (job === "calibrate" && r.scores) {
    const m = r.scores;
    const note = r.seen_in_training ? " (the network trained on this session, optimistic)" : "";
    const wrist = m.wrist_as_sign != null;
    return `<tr><th>other repetitions${note}</th><th>balanced</th><th>rest taken for a position</th>
      <th>wrong position shown</th>${wrist ? "<th>wrist bends read as a sign</th>" : ""}</tr>
      <tr><td>${r.saved}</td><td>${pct(m.balanced)}</td><td>${pct(m.rest_false)}</td><td>${pct(m.wrong)}</td>
      ${wrist ? `<td>${pct(m.wrist_as_sign)}</td>` : ""}</tr>`;
  }
  return "";
}

const PREDICTED_COLOR = cssVar("--predicted");
const EFFORT_COLOR = cssVar("--effort");
const netView = { yaw: -45, pitch: -25 };

// The network predicts the base and middle joints only, the others come from the relaxed pose
function predictedHand(angles) {
  return { ...anglesByName(config.gestures.rest.angles), ...angles };
}

// One view: the camera's hand with one head's prediction over it, and its match score
function drawHeadHand(canvas, match, p, predicted, color, missing, name) {
  const { ctx } = sharpen($(canvas), false);
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  const tracked = state.hand ? anglesByName(state.hand.angles) : null;
  if (tracked) drawHand(ctx, tracked, { view: netView, mirror: mirrored(), color: TRACKED_COLOR, width: 5 });
  if (predicted) {
    drawHand(ctx, predictedHand(predicted), { view: netView, mirror: mirrored(), color, width: 7, alpha: 0.85 });
  }
  let html = "";
  if (!p.running) html = "<small>no network running</small>";
  else if (!predicted) html = `<small>${missing}</small>`;
  else if (!tracked) html = "<small>no hand in view, nothing to compare against</small>";
  else {
    // Same score as the Record tab's pose match
    const joints = Object.keys(predicted);
    const mean = joints.reduce((sum, j) => sum + Math.abs(predicted[j] - tracked[j]), 0) / joints.length;
    html = `<span style="color:${color}">${matchScore(mean)} %</span> match, ${name}<small>mean error ${mean.toFixed(0)}° on ${joints.length} joints</small>`;
  }
  $(match).innerHTML = html;
}

function drawNetHand(p) {
  document.querySelector(".hand-pair").classList.toggle("idle", !p.running);
  const lda = "this model doesn't predict finger angles (LDA)";
  drawHeadHand("net-hand", "net-match", p, p.running ? p.angles : null, PREDICTED_COLOR, lda, "angle head");
  drawHeadHand("net-hand-effort", "net-match-effort", p, p.running ? p.effort_angles : null, EFFORT_COLOR,
    p.angles ? "this network has no effort head, train a new one" : lda, "effort head");
}

for (const id of ["net-hand", "net-hand-effort"]) {
  attachRotate($(id), netView, () => state?.tools && drawNetHand(state.tools.predictor));
}

// Same setting as the hand on the Record tab: which hand the camera tracks and how the hands are drawn
$("net-side").addEventListener("change", () => {
  const field = $("setup").tracked_hand;
  field.value = $("net-side").value;
  field.dispatchEvent(new Event("input", { bubbles: true }));
});

let networkWasRunning = false;
let shownNetError = null;
function drawNetwork(n, p) {
  for (const id of ["net-eval-newest", "net-eval", "net-calibrate"]) $(id).disabled = n.running;
  $("train-start").disabled = n.running || state.tools.trainer.running;
  $("net-stop").disabled = !n.running;
  // Only there while a network job runs, next to Train it read as part of LDA training
  $("net-stop").hidden = !n.running;
  // Time since the last line tells a slow job from a stalled one, every finished epoch prints a line
  const pr = n.progress || {};
  const where = [pr.fold && `fold ${pr.fold[0]} of ${pr.fold[1]}`, pr.epoch && `epoch ${pr.epoch[0]} of ${pr.epoch[1]}`,
    !pr.fold && pr.left && `about ${pr.left} left`].filter(Boolean).join(", ");
  $("net-step").textContent = n.running
    ? `${n.job.replace("_", " ")} running for ${fmtTime(n.running_s ?? 0)}${where ? `, ${where}` : ""}, last output ${fmtTime(n.quiet_s ?? 0)} ago`
    : "";
  $("net-progress").hidden = !n.running;
  $("net-bar").classList.toggle("waiting", n.running && pr.share == null);
  $("net-bar").style.width = n.running && pr.share != null ? `${100 * pr.share}%` : "";
  // Only a new error from the server is shown. An update sent just before a click would otherwise put the last
  // job's error back after the click cleared it
  if (n.error && n.error !== shownNetError) $("net-error").textContent = n.error;
  shownNetError = n.error;
  // Only the last lines matter while it runs, the table replaces them once there is a result
  const html = n.result ? networkTable(n.job, n.result) : "";
  if ($("net-table").innerHTML !== html) $("net-table").innerHTML = html;
  $("net-table").hidden = !html;
  const lines = n.lines || [];
  $("net-log-box").hidden = !lines.length;
  $("net-log-last").textContent = lines.length ? lines[lines.length - 1].trim() : "";
  showLog($("net-lines"), lines);
  if (networkWasRunning && !n.running && currentTab === "train") loadTrainLists();
  networkWasRunning = n.running;
  const side = $("setup").tracked_hand.value;
  if (document.activeElement !== $("net-side") && $("net-side").value !== side) $("net-side").value = side;
  drawNetHand(p);
}

// Connection check

$("conn-start").addEventListener("click", () => postShowing("/api/check", undefined, "conn-error"));
$("wifi-start").addEventListener("click", () => {
  postShowing("/api/armband/wifi", { dhcp_s: Number($("wifi-dhcp").value) || 0 }, "wifi-error");
});

function drawWifi(t) {
  $("wifi-start").disabled = t.running;
  if (t.error) $("wifi-error").textContent = t.error;
  $("wifi-result").textContent = (t.lines || []).join("\n") + (t.running ? "\n..." : "");
}

function drawConnection(t) {
  $("conn-start").disabled = t.running;
  progress("conn-bar", t);
  if (t.error) $("conn-error").textContent = t.error;
  $("conn-result").textContent = (t.lines || []).join("\n");
  const cam = config.camera ? `camera at ${state.camera_fps ?? 0} fps` : "no camera";
  $("conn-source").textContent = `EMG from ${config.source}, ${config.channels.length} channels at ${config.rate} Hz, ${cam}`;
}

// Same thresholds as the warning bar (quality.BATTERY_LOW) plus an earlier yellow step
const BATTERY_LOW = 30, BATTERY_CRITICAL = 15;
// A full three-posture session with its breaks, for the "sessions left" estimate
const SESSION_MINUTES = 14;

function drawBattery(b) {
  const chip = $("battery-chip");
  chip.hidden = !b;
  if (!b) {
    $("battery-detail").innerHTML = `<div>none<small>${config.source} has no battery reading</small></div>`;
    return;
  }
  chip.textContent = `Battery ${Math.round(b.percent)} %`;
  chip.classList.toggle("low", b.percent < BATTERY_LOW && b.percent >= BATTERY_CRITICAL);
  chip.classList.toggle("critical", b.percent < BATTERY_CRITICAL);
  const rate = b.per_min != null ? `${b.per_min.toFixed(2)} %` : "measuring";
  const left = b.minutes_left != null ? `${Math.floor(b.minutes_left / 60)} h ${b.minutes_left % 60} min` : "measuring";
  const sessions = b.minutes_left != null ? `${Math.floor(b.minutes_left / SESSION_MINUTES)}` : "measuring";
  $("battery-detail").innerHTML = `<div>${Math.round(b.percent)} %<small>level</small></div>
    <div>${rate}<small>used per minute</small></div><div>${left}<small>left at that rate</small></div>
    <div>${sessions}<small>full sessions (${SESSION_MINUTES} min each)</small></div>`;
}

function drawTools() {
  const t = state.tools;
  drawBattery(state.battery);
  $("rec-badge").hidden = !state.session;
  $("rec-badge").textContent = state.session?.practice ? "practice, not saved" : "recording";
  $("rec-badge").classList.toggle("practice", !!state.session?.practice);
  if (!t) return;
  drawCalibration(t.calibration);
  drawHaptics(t.haptics, t.test);
  drawConnection(t.check);
  drawWifi(t.wifi);
  drawTrain(t.trainer);
  drawPredictor(t.predictor);
  drawNetwork(t.network, t.predictor);
  drawControl(t.control, t.predictor);
  drawCursor(t.cursor);
}

function initTools() {
  let link = {};
  try { link = JSON.parse(localStorage.getItem(LINK_KEY)) || {}; } catch { link = {}; }
  $("esp-link").value = link.mode || "auto";
  $("esp-port").value = link.port || "";
  $("esp-ip").value = link.ip || config.esp32_wifi_ip;
  saveLink();
  showCurrentCalibration();
}
