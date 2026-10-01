// Tabs for the existing tools: live signals (live_plot.py), calibration (calibrate.py), the gesture model
// (ml/gesture_model.py), haptics and the output test (haptics.py, esp32_link.py) and the connection check
// (check_connection.py).

const LINK_KEY = "esp32-link";
const SPECTRUM_MS = 250;

// Tabs

function showTab(name) {
  currentTab = name;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => { t.hidden = t.id !== `tab-${name}`; });
  // The camera preview and the signal stream only run while their tab is open
  if (name === "record") showView();
  else $("video").src = "";
  if (name === "signals") openSignals();
  else closeSignals();
  if (name === "train") loadTrainLists();
}

document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

function progress(bar, tool) {
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
  const { width: w, height: h } = ctx.canvas;
  ctx.clearRect(0, 0, w, h);
  const rowH = h / config.channels.length;
  ctx.font = "13px system-ui";
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
    ctx.fillStyle = "#e6e6e6";
    ctx.fillText(`ch ${config.channels[c]}`, 8, mid - 4);
    ctx.fillStyle = "#8b919c";
    ctx.fillText(`rms ${count ? Math.sqrt(sq / count).toFixed(1) : "-"} µV`, 8, mid + 13);
    ctx.strokeStyle = "#2f333b";
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
  ctx.fillStyle = "#8b919c";
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
    ctx.fillStyle = "#e6e6e6";
    ctx.fillText(`ch ${config.channels[c]}`, 8, top + rowH / 2);
    // Mains and its harmonics, where the notches sit
    ctx.strokeStyle = "#3a2f2f";
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
  ctx.fillStyle = "#8b919c";
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
  if (e.key === "f") toggleSignal("sig-filtered");
  if (e.key === "s") toggleSignal("sig-spectrum");
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

$("cal-start").addEventListener("click", async () => {
  $("cal-error").textContent = "";
  try {
    await post("/api/calibrate", { seconds: Number($("cal-seconds").value), car: $("cal-car").checked });
  } catch (err) {
    $("cal-error").textContent = err.message;
  }
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

async function hapticsCommand(path, body) {
  $("hap-error").textContent = "";
  try {
    await post(path, body);
  } catch (err) {
    $("hap-error").textContent = err.message;
  }
}

$("hap-start").addEventListener("click", () => hapticsCommand("/api/haptics/start", linkBody()));
$("hap-stop").addEventListener("click", () => hapticsCommand("/api/haptics/stop"));
$("esp-test").addEventListener("click", () => hapticsCommand("/api/esp32/test", linkBody()));

function drawOutputs(h) {
  const ctx = $("outputs").getContext("2d");
  const { width: w, height: hgt } = ctx.canvas;
  ctx.clearRect(0, 0, w, hgt);
  const outputs = h.outputs || [];
  if (!outputs.length) return;
  const slot = w / outputs.length;
  const top = 10, bottom = 44;
  const plotH = hgt - top - bottom;
  ctx.font = "13px system-ui";
  outputs.forEach((chs, i) => {
    const duty = h.duties?.[i] ?? 0;
    const x = i * slot + slot * 0.15;
    const bw = slot * 0.7;
    ctx.fillStyle = "#2a2e36";
    ctx.fillRect(x, top, bw, plotH);
    const lost = chs.some((ch) => h.lost?.[config.channels.indexOf(ch)]);
    ctx.fillStyle = lost ? "#e5484d" : TRACKED_COLOR;
    ctx.fillRect(x, top + plotH * (1 - duty / 100), bw, plotH * duty / 100);
    ctx.fillStyle = "#e6e6e6";
    ctx.fillText(`out ${i}  ${duty}%`, x, hgt - bottom + 18);
    ctx.fillStyle = "#8b919c";
    ctx.fillText(`ch ${chs.join("+")}`, x, hgt - bottom + 35);
  });
}

function drawHaptics(h, test) {
  $("hap-start").disabled = h.running || test.running;
  $("hap-stop").disabled = !h.running;
  $("esp-test").disabled = h.running || test.running;
  if (h.running) {
    $("hap-status").textContent = h.link
      ? `Running on ${h.link}, common average ${h.car ? "on" : "off"}, watchdog fired ${h.watchdog} times`
      : "Connecting to the ESP32...";
  } else {
    $("hap-status").textContent = test.running ? "Output test running..." : "Stopped";
  }
  if (h.error || test.error) $("hap-error").textContent = h.error || test.error;
  drawOutputs(h);
  $("esp-log").textContent = (test.lines || []).join("\n");
}

// Gesture model: training and live test

async function loadTrainLists() {
  const [sessions, models] = await Promise.all([fetch("/api/sessions"), fetch("/api/models")].map((r) => r.then((x) => x.json())));
  const table = $("train-sessions");
  const checked = new Set([...table.querySelectorAll("input:checked")].map((c) => c.value));
  const listed = new Set([...table.querySelectorAll("input")].map((c) => c.value));
  const usable = sessions.filter((s) => s.gestures > 0);
  table.innerHTML = "<tr><th></th><th>Session</th><th>Arm</th><th>Postures</th><th>Placement</th><th>Length</th><th>Camera delay</th></tr>";
  for (const s of usable) {
    const tr = document.createElement("tr");
    // Keep the choice for sessions already listed, a session recorded since then starts ticked if it finished
    const on = listed.has(s.name) ? checked.has(s.name) : s.completed;
    tr.innerHTML = `<td><input type="checkbox" value="${s.name}" ${on ? "checked" : ""}></td><td>${s.name}</td>
      <td>${s.band_arm ?? ""}</td><td>${(s.postures || []).join(", ")}</td><td>${s.placement || ""}</td>
      <td>${s.seconds ? fmtTime(s.seconds) : ""}${s.completed ? "" : " (stopped early)"}</td>
      <td>${s.camera_delay_s != null ? `${Math.round(1000 * s.camera_delay_s)} ms` : "-"}</td>`;
    table.append(tr);
  }
  if (!usable.length) table.innerHTML = "<tr><td>No sessions with gestures yet, record one on the Record tab.</td></tr>";
  const select = $("model-select");
  const current = select.value;
  select.innerHTML = models.map((m) => {
    const acc = m.accuracy != null ? `, ${Math.round(100 * m.accuracy)} % on unseen sessions` : "";
    // Session folders start with date and time, that's enough to tell them apart
    const from = m.sessions.map((s) => s.slice(0, 17).replace("_", " ")).join(", ");
    return `<option value="${m.name}">${m.name}, trained on ${from}${acc}</option>`;
  }).join("") || "<option value=''>no models yet</option>";
  // A model that was just trained is already running, show that one
  const running = state?.tools?.predictor?.running && state.tools.predictor.model;
  if (running && models.some((m) => m.name === running)) select.value = running;
  else if (current && models.some((m) => m.name === current)) select.value = current;
}

$("train-start").addEventListener("click", async () => {
  $("train-error").textContent = "";
  const sessions = [...$("train-sessions").querySelectorAll("input:checked")].map((c) => c.value);
  try {
    await post("/api/train", {
      sessions, log: $("train-log").checked, accel: $("train-accel").checked, vote: Number($("train-vote").value),
    });
  } catch (err) {
    $("train-error").textContent = err.message;
  }
});

$("model-use").addEventListener("click", async () => {
  $("model-error").textContent = "";
  try {
    await post("/api/model/use", { name: $("model-select").value });
  } catch (err) {
    $("model-error").textContent = err.message;
  }
});
$("model-stop").addEventListener("click", () => post("/api/model/stop").catch(() => {}));
$("model-threshold").addEventListener("change", () => {
  const value = Math.min(0.99, Math.max(0, Number($("model-threshold").value)));
  post("/api/model/threshold", { value }).catch((err) => { $("model-error").textContent = err.message; });
});

let trainWasRunning = false;
function drawTrain(t) {
  $("train-start").disabled = t.running;
  $("train-step").textContent = t.running ? `${t.step || "starting"}...` : "";
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

function drawProbs(p) {
  const ctx = $("live-probs").getContext("2d");
  const { width: w, height: h } = ctx.canvas;
  ctx.clearRect(0, 0, w, h);
  if (!p.running || !p.probs) return;
  const classes = p.classes;
  const rowH = h / classes.length;
  const left = 110;
  ctx.font = "14px system-ui";
  classes.forEach((c, i) => {
    const v = p.probs[c] ?? 0;
    const y = i * rowH;
    ctx.fillStyle = c === p.label ? "#e6e6e6" : "#8b919c";
    ctx.fillText(c.replace("_", " "), 4, y + rowH * 0.65);
    ctx.fillStyle = "#2a2e36";
    ctx.fillRect(left, y + 3, w - left, rowH - 6);
    ctx.fillStyle = c === p.label ? TRACKED_COLOR : "#4b5563";
    ctx.fillRect(left, y + 3, (w - left) * v, rowH - 6);
  });
  // A gesture's bar has to cross this line before it is shown
  const x = left + (w - left) * (p.threshold ?? 0);
  ctx.strokeStyle = "#e0a02a";
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
}

// The virtual hand shows the predicted gesture's target pose
function drawModelView() {
  const p = state.tools?.predictor;
  const ctx = $("model-hand").getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  if (!p?.running) {
    $("model-label").innerHTML = "<small>no model running, train or pick one on the Train tab</small>";
    return;
  }
  $("model-label").innerHTML = predictionLabel(p);
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
  drawProbs(p);
  // Rewriting an option redraws an open dropdown, so only touch it when the model starts or stops
  const option = $("model-option");
  if (option.disabled === !!p.running) {
    option.disabled = !p.running;
    option.textContent = p.running ? "Model prediction" : "Model prediction (no model running)";
  }
}

// Connection check

$("conn-start").addEventListener("click", async () => {
  $("conn-error").textContent = "";
  try {
    await post("/api/check");
  } catch (err) {
    $("conn-error").textContent = err.message;
  }
});

function drawConnection(t) {
  $("conn-start").disabled = t.running;
  progress("conn-bar", t);
  if (t.error) $("conn-error").textContent = t.error;
  $("conn-result").textContent = (t.lines || []).join("\n");
  const cam = config.camera ? `camera at ${state.camera_fps ?? 0} fps` : "no camera";
  $("conn-source").textContent = `EMG from ${config.source}, ${config.channels.length} channels at ${config.rate} Hz, ${cam}`;
}

function drawTools() {
  const t = state.tools;
  $("rec-badge").hidden = !state.session;
  $("rec-badge").textContent = state.session?.practice ? "practice, not saved" : "recording";
  $("rec-badge").classList.toggle("practice", !!state.session?.practice);
  if (!t) return;
  drawCalibration(t.calibration);
  drawHaptics(t.haptics, t.test);
  drawConnection(t.check);
  drawTrain(t.trainer);
  drawPredictor(t.predictor);
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
