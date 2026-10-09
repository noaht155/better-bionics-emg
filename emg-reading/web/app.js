// Recording page. The server sends the whole live state about 20 times a second over /ws,
// the page only draws it and posts button presses back.

const $ = (id) => document.getElementById(id);
const SETTINGS_KEY = "recorder-settings";
const SERIOUS = new Set(["no_data", "contact", "hum", "camera", "loop_error"]);
const TARGET_COLOR = "#6b7280";
const TRACKED_COLOR = "#4fa3ff";

let config = null;
let state = null;
let currentTab = "record";
// Seen a little from the little-finger side, where curled fingers and the thumb stay readable
const view = { yaw: -45, pitch: -25 };
// The filled hand reads best with the palm towards the viewer, turned a little
const solidView = { yaw: 25, pitch: -20 };
const SOLID_COLOR = "#c8cdd6";
const UPCOMING_COLOR = "#6f7682";

function anglesByName(list) {
  return Object.fromEntries(config.joints.map((j, i) => [j, list[i]]));
}

// Hands are drawn like a mirror, the same as the camera preview: a left hand's reflection has the shape of a right
// hand, so the left hand is drawn unmirrored and the right hand mirrored
function mirrored() {
  return $("setup").tracked_hand.value === "right";
}

// Setup form

function loadSettings() {
  const form = $("setup");
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(SETTINGS_KEY)) || {}; } catch { saved = {}; }
  for (const [name, text] of Object.entries(config.postures)) {
    const label = document.createElement("label");
    label.innerHTML = `<input type="checkbox" name="posture" value="${name}" checked> ${text}`;
    $("postures").append(label);
  }
  for (const [k, v] of Object.entries(saved)) {
    if (k === "postures") {
      form.querySelectorAll("[name=posture]").forEach((c) => { c.checked = v.includes(c.value); });
    } else if (form[k] && typeof v !== "object") {
      form[k].value = v;
    }
  }
  // The server starts with its default hand after every restart, so the saved choice is sent to it. During a
  // session it keeps its own and the form follows
  const hand = saved.tracked_hand || config.tracked_hand;
  if (hand) {
    form.tracked_hand.value = hand;
    if (hand !== config.tracked_hand) {
      post("/api/hand", { hand }).then((r) => { if (r.hand) form.tracked_hand.value = r.hand; }).catch(() => {});
    }
  }
  showProtocolFields();
  updateDuration();
}

function readSettings() {
  const form = $("setup");
  const num = (name) => Number(form[name].value);
  return {
    subject: form.subject.value.trim(),
    band_arm: form.band_arm.value,
    port_facing: form.port_facing.value,
    tracked_hand: form.tracked_hand.value,
    placement: form.placement.value.trim(),
    notes: form.notes.value.trim(),
    postures: [...form.querySelectorAll("[name=posture]:checked")].map((c) => c.value),
    protocol: form.protocol.value,
    reps: num("reps"), hold_s: num("hold_s"), rest_s: num("rest_s"), rounds: num("rounds"), move_s: num("move_s"),
    practice: form.practice.checked,
  };
}

function planBody() {
  const { postures, protocol, reps, hold_s, rest_s, rounds, move_s } = readSettings();
  return { postures, protocol, reps, hold_s, rest_s, rounds, move_s };
}

// Only the fields the chosen protocol uses
function showProtocolFields() {
  const form = $("setup");
  form.querySelectorAll("[data-protocol]").forEach((el) => { el.hidden = el.dataset.protocol !== form.protocol.value; });
}

// Matches gestures.session_plan, without the breaks that wait for Continue
// The server works out the length from the real session plan
async function updateDuration() {
  const s = readSettings();
  if (!s.postures.length) {
    $("duration").textContent = "Tick at least one posture";
    return;
  }
  try {
    const plan = await post("/api/plan", planBody());
    $("duration").textContent = `About ${Math.round(plan.seconds / 60)} min plus breaks between postures`;
  } catch {
    $("duration").textContent = "";
  }
}

async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const detail = (await res.json().catch(() => ({}))).detail;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

$("setup").addEventListener("input", (e) => {
  // Protocol B is normally done with the forearm on the table only, like Ninapro. Other postures can still be ticked
  if (e.target.name === "protocol" && e.target.value === "B") {
    $("setup").querySelectorAll("[name=posture]").forEach((c) => { c.checked = c.value === "table"; });
  }
  showProtocolFields();
  updateDuration();
  if (e.target.name === "tracked_hand") {
    post("/api/hand", { hand: e.target.value }).catch(() => {});
    // Saved straight away, not only when a session starts, so it survives a restart
    try {
      const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY)) || {};
      localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...saved, tracked_hand: e.target.value }));
    } catch { /* private window */ }
  }
  $("start-button").textContent = $("practice").checked ? "Start practice run" : "Start recording";
});

$("setup").addEventListener("submit", async (e) => {
  e.preventDefault();
  const settings = readSettings();
  $("setup-error").textContent = "";
  // Practice isn't remembered, a forgotten tick would quietly throw away a real session
  const { practice, ...remembered } = settings;
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(remembered)); } catch { /* private window */ }
  try {
    await post("/api/session/start", settings);
  } catch (err) {
    $("setup-error").textContent = err.message;
  }
});

// Cue preview: every distinct cue of the planned session once, to go through with the person before recording

let preview = null;

async function openPreview() {
  $("setup-error").textContent = "";
  try {
    const { plan } = await post("/api/plan", planBody());
    const seen = new Set();
    const cues = plan.filter((c) => !seen.has(c.text) && seen.add(c.text));
    preview = { cues, at: 0, shownAt: performance.now() };
  } catch (err) {
    $("setup-error").textContent = err.message;
    return;
  }
  $("setup").hidden = true;
  $("preview").hidden = false;
  drawPreview();
}

function closePreview() {
  preview = null;
  $("preview").hidden = true;
  $("setup").hidden = !!state?.session;
}

function stepPreview(by) {
  if (!preview) return;
  preview.at = Math.min(Math.max(preview.at + by, 0), preview.cues.length - 1);
  preview.shownAt = performance.now();
}

function drawPreview(dt = 0) {
  const cue = preview.cues[preview.at];
  const posture = config.postures[cue.posture] ?? cue.posture;
  $("preview-info").textContent = cue.kind === "break" ? "waits for Continue" : `${cue.seconds} s, ${posture}`;
  $("preview-count").textContent = `${preview.at + 1} / ${preview.cues.length}`;
  $("preview-text").innerHTML = cueTitle(cue.text);
  const ctx = $("preview-hand").getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  // Plays on a loop at the real pace from the moment the cue is shown, the hand easing over from the last one
  const t = ((performance.now() - preview.shownAt) / 1000) % (cue.seconds || 4);
  const angles = follow("preview", cue.kind === "break" ? null : cueFrame(cue, t), dt);
  if (angles) drawSolidHand(ctx, angles, { view: solidView, mirror: mirrored(), color: SOLID_COLOR });
}

attachRotate($("preview-hand"), solidView, () => preview && drawPreview());
$("preview-button").addEventListener("click", openPreview);
$("preview-close").addEventListener("click", closePreview);
$("preview-prev").addEventListener("click", () => stepPreview(-1));
$("preview-next").addEventListener("click", () => stepPreview(1));
document.addEventListener("keydown", (e) => {
  if (!preview || currentTab !== "record") return;
  if (e.key === "ArrowLeft") stepPreview(-1);
  if (e.key === "ArrowRight") stepPreview(1);
  if (e.key === "Escape") closePreview();
});

// Session controls

function command(action) {
  post(`/api/session/${action}`).catch((err) => console.warn(action, err.message));
}

function mainAction() {
  const s = state?.session;
  if (!s) return;
  command(s.paused || s.cue.kind === "break" ? "continue" : "pause");
}

$("btn-main").addEventListener("click", mainAction);
$("btn-bad").addEventListener("click", () => command("bad"));

// Stop needs a second click within 3 s, so a stray click doesn't end the session
let stopArmed = null;
$("btn-stop").addEventListener("click", () => {
  if (stopArmed) {
    clearTimeout(stopArmed);
    stopArmed = null;
    $("btn-stop").classList.remove("armed");
    $("btn-stop").textContent = "Stop";
    command("stop");
    return;
  }
  $("btn-stop").classList.add("armed");
  $("btn-stop").textContent = "Click again to stop";
  stopArmed = setTimeout(() => {
    stopArmed = null;
    $("btn-stop").classList.remove("armed");
    $("btn-stop").textContent = "Stop";
  }, 3000);
});

document.addEventListener("keydown", (e) => {
  if (!state?.session || currentTab !== "record" || e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); mainAction(); }
  if (e.key === "b") command("bad");
});

// Drawing

function fmtTime(s) {
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

function cueTitle(text) {
  const [first, ...rest] = text.split("\n");
  return rest.length ? `${first}<small>${rest.join("<br>")}</small>` : first;
}

function drawWarnings() {
  const box = $("warnings");
  box.innerHTML = "";
  for (const w of state.warnings) {
    const div = document.createElement("div");
    div.textContent = w.text;
    if (SERIOUS.has(w.code)) div.className = "serious";
    box.append(div);
  }
}

function drawSession() {
  const s = state.session;
  if (s) preview = null;
  $("setup").hidden = !!s || !!preview;
  $("preview").hidden = !preview;
  $("cue-panel").hidden = !s;
  if (!s) {
    drawSummary();
    return;
  }
  const cue = s.cue;
  $("posture").textContent = config.postures[cue.posture] ?? cue.posture;
  $("progress-text").textContent = `cue ${s.index + 1} of ${s.count}   ${fmtTime(s.elapsed)}`;
  const hum = s.hum_pause;
  const pausedText = hum
    ? `HUM ON CH ${hum.channels.join(", ")}<small>${hum.uv.map((v) => `${v} uV`).join(", ")}, that part is marked bad.
      Press ${hum.channels.length > 1 ? "those pads" : "that pad"} down firmly, then Continue (space) to redo it</small>`
    : "PAUSED<small>press Continue (space) to redo this cue</small>";
  $("cue-text").innerHTML = s.paused ? pausedText : cueTitle(cue.text);
  $("cue-text").classList.toggle("paused", s.paused);
  if (cue.seconds) {
    const left = Math.max(0, cue.seconds - s.cue_elapsed);
    $("cue-bar").style.width = `${Math.min(100, (100 * s.cue_elapsed) / cue.seconds)}%`;
    $("cue-time").textContent = s.paused ? "" : `${left.toFixed(1)} s`;
  } else {
    $("cue-bar").style.width = "0";
    $("cue-time").textContent = "";
  }
  // Only when the hand below shows the next gesture, so the text and the picture always agree. Name and
  // description, without the reminder line every gesture shares
  const target = targetPose();
  $("next").innerHTML = target?.upcoming ? `<span>Next</span>${cueTitle(target.text)}` : "";
  $("btn-main").textContent = s.paused || cue.kind === "break" ? "Continue (space)" : "Pause (space)";
  $("session-info").textContent = `${s.folder}   ${s.samples} EMG samples, ${s.frames} camera frames`;

  const caption = $("target-label");
  caption.textContent = target ? `${target.upcoming ? "Next" : "Now"}: ${target.title}` : "";
  caption.classList.toggle("upcoming", !!target?.upcoming);
}

// The drawn hands ease towards their target instead of jumping, so a change of cue looks like one hand moving.
// They redraw every screen frame; between the server's updates (20 a second) the cue time runs on locally
const EASE_S = 0.15;
const shown = {};
let stateAt = 0;
let lastFrame = performance.now();

function follow(key, angles, dt) {
  if (!angles) return (shown[key] = null);
  const cur = shown[key];
  if (!cur) return (shown[key] = { ...angles });
  const k = 1 - Math.exp(-dt / EASE_S);
  for (const j in angles) cur[j] += k * (angles[j] - cur[j]);
  return cur;
}

function drawTarget(dt) {
  const s = state?.session;
  const ctx = $("target").getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  if (!s) return follow("target", null);
  const elapsed = s.cue_elapsed + (s.paused ? 0 : (performance.now() - stateAt) / 1000);
  const target = targetPose(elapsed);
  const angles = follow("target", target?.angles, dt);
  // The next gesture is drawn darker so it doesn't read as the current one. Not transparent, the overlapping
  // parts would show through each other
  if (angles) {
    drawSolidHand(ctx, angles, { view: solidView, mirror: mirrored(), color: target.upcoming ? UPCOMING_COLOR : SOLID_COLOR });
  }
}

function frameLoop(now) {
  const dt = Math.min((now - lastFrame) / 1000, 0.1);
  lastFrame = now;
  try {
    drawTarget(dt);
    if (preview) drawPreview(dt);
  } catch (err) {
    console.error("frameLoop", err);
  }
  requestAnimationFrame(frameLoop);
}

function drawSummary() {
  const box = $("summary");
  const r = state.summary;
  box.hidden = !r;
  if (!r) return;
  const detected = r.hand_detected == null ? "no camera" : `${Math.round(100 * r.hand_detected)} %`;
  const what = r.practice ? "Practice run (nothing saved)" : "Last session";
  box.innerHTML = `<b>${what} ${r.completed ? "finished" : "stopped early"}</b><br>
    ${r.practice ? "" : `${r.folder}<br>`}${fmtTime(r.seconds)}, ${r.samples} EMG samples, ${r.dropped_samples} dropped,
    ${r.frames} camera frames, hand detected in ${detected}`;
}

const FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"];
const clamp01 = (x) => Math.min(Math.max(x, 0), 1);
const ease = (x) => (1 - Math.cos(Math.PI * clamp01(x))) / 2;

// The open hand plus a share of the way to each named pose: [[label, share], ...]
function blendPoses(parts) {
  const open = config.gestures.open.angles;
  const out = open.slice();
  for (const [label, f] of parts) {
    config.gestures[label].angles.forEach((a, i) => { out[i] += f * (a - open[i]); });
  }
  return anglesByName(out);
}

// One finger cue cycle as its text says: bend halfway, pause, bend fully, back
function fingerShare(x) {
  if (x < 0.25) return 0.5 * ease(x / 0.25);
  if (x < 0.4) return 0.5;
  if (x < 0.65) return 0.5 + 0.5 * ease((x - 0.4) / 0.25);
  return 1 - ease((x - 0.65) / 0.35);
}

// Where the target hand is t seconds into a cue, so the person can follow it: a protocol B movement closes and
// opens once, a finger cue does its three bends, the wave bends the fingers one after another, a grip closes from
// the open hand in half a second and then holds
function cueFrame(cue, t) {
  const T = cue.seconds || 4;
  if (cue.kind === "move") return blendPoses([[cue.label, (1 - Math.cos(2 * Math.PI * clamp01(t / T))) / 2]]);
  if (cue.label === "wave") {
    const cycle = 2;
    return blendPoses(FINGER_NAMES.slice(1).map((f, k) => {
      const x = ((t / cycle - k / 4) % 1 + 1) % 1;
      return [`flex_${f}`, x < 0.5 ? (1 - Math.cos(4 * Math.PI * x)) / 2 : 0];
    }));
  }
  if (!config.gestures[cue.label]) return null;
  if (cue.kind === "finger") return blendPoses([[cue.label, fingerShare((t % (T / 3)) / (T / 3))]]);
  if (cue.kind === "hold" && cue.label !== "rest") return blendPoses([[cue.label, ease(t / 0.5)]]);
  return anglesByName(config.gestures[cue.label].angles);
}

function cueAngles(cue, s, elapsed) {
  const g = config.gestures[cue.label];
  // The next cue, shown during a rest, is drawn at its end pose so it can be prepared. A grip is shown that way
  // already, so during its own cue it simply holds; the eased hand makes the change from rest
  if (s.cue !== cue || cue.kind === "hold") return g ? anglesByName(g.angles) : null;
  return cueFrame(cue, elapsed);
}

// Target pose of the current cue, or the next gesture during a rest so it can be prepared
// Returns {angles, title, text, upcoming}, upcoming is true when it shows the next gesture
function targetPose(elapsed) {
  const s = state.session;
  if (!s) return null;
  let cue = s.cue;
  if (cue.kind === "free" || cue.kind === "sync" || cue.kind === "break") return null;
  const upcoming = cue.label === "rest" && s.next && !!config.gestures[s.next.label];
  if (upcoming) cue = s.next;
  const angles = cueAngles(cue, s, elapsed ?? s.cue_elapsed);
  const [title, description] = cue.text.split("\n");
  return angles ? { angles, title, text: `${title}\n${description ?? ""}`.trim(), upcoming } : null;
}

function drawPose() {
  const ctx = $("pose").getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  const s = state.session;
  const cue = s?.cue;
  const g = cue && ["hold", "finger", "move"].includes(cue.kind) ? config.gestures[cue.label] : null;
  const target = g ? cueAngles(cue, s) : null;
  const tracked = state.hand ? anglesByName(state.hand.angles) : null;
  if (target) drawHand(ctx, target, { view, mirror: mirrored(), color: TARGET_COLOR, width: 9, alpha: 0.6 });
  if (tracked) drawHand(ctx, tracked, { view, mirror: mirrored(), color: TRACKED_COLOR, width: 5 });
  let html = "";
  if (!tracked) {
    html = "<small>no hand in view</small>";
  } else if (target) {
    const errors = config.reliable.map((j) => Math.abs(tracked[j] - target[j]));
    const mean = errors.reduce((a, b) => a + b, 0) / errors.length;
    html = `${Math.max(0, Math.round(100 - 2 * mean))} %<small>mean error ${mean.toFixed(0)}° on the base and middle joints</small>`;
  } else {
    html = "<small>tracked hand</small>";
  }
  $("score").innerHTML = html;
}

// Log scale, resting EMG is a few uV and a hard clench several hundred
const EMG_MIN = 1, EMG_MAX = 1000;
function emgY(uv, top, height) {
  const t = Math.log10(Math.max(EMG_MIN, uv) / EMG_MIN) / Math.log10(EMG_MAX / EMG_MIN);
  return top + height * (1 - Math.min(1, t));
}

function drawEmg() {
  const ctx = $("emg").getContext("2d");
  const { width: w, height: h } = ctx.canvas;
  ctx.clearRect(0, 0, w, h);
  const top = 10, bottom = 50, left = 40;
  const plotH = h - top - bottom;
  ctx.font = "12px system-ui";
  ctx.fillStyle = "#8b919c";
  ctx.strokeStyle = "#2f333b";
  for (const uv of [1, 10, 100, 1000]) {
    const y = emgY(uv, top, plotH);
    ctx.fillText(`${uv}`, 4, y + 4);
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(w, y); ctx.stroke();
  }
  const n = config.channels.length;
  const slot = (w - left) / n;
  config.channels.forEach((ch, i) => {
    const x = left + i * slot + slot * 0.15;
    const bw = slot * 0.7;
    const y = emgY(state.env[i] ?? 0, top, plotH);
    ctx.fillStyle = state.lost[i] ? "#e5484d" : "#4fa3ff";
    ctx.fillRect(x, y, bw, top + plotH - y);
    const cal = config.calibration?.[String(ch)];
    if (cal) {
      ctx.fillStyle = "#e6e6e6";
      for (const uv of [cal.rest, cal.max]) ctx.fillRect(x - 3, emgY(uv, top, plotH) - 1, bw + 6, 2);
    }
    ctx.fillStyle = "#e6e6e6";
    ctx.fillText(`ch ${ch}`, x, h - bottom + 16);
    ctx.fillStyle = "#8b919c";
    ctx.fillText(`${Math.round(state.env[i] ?? 0)} µV`, x, h - bottom + 31);
    ctx.fillText(`hum ${Math.round(state.hum[i] ?? 0)}`, x, h - bottom + 46);
  });
}

// Landmark pairs of MediaPipe's hand skeleton
const LINKS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

function drawCamera() {
  const img = $("video");
  const canvas = $("overlay");
  const ctx = canvas.getContext("2d");
  // Firefox reports a size of 0 between the frames of the video stream. Clear first and keep the last size,
  // otherwise the old skeleton stays drawn after the hand is lost
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (img.naturalWidth && canvas.width !== img.naturalWidth) {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
  }
  if (!state.hand) return;
  const pts = state.hand.image.map(([x, y]) => [x * canvas.width, y * canvas.height]);
  ctx.strokeStyle = TRACKED_COLOR;
  ctx.lineWidth = 3;
  for (const [a, b] of LINKS) {
    ctx.beginPath(); ctx.moveTo(...pts[a]); ctx.lineTo(...pts[b]); ctx.stroke();
  }
}

function drawView() {
  const v = $("view").value;
  if (v === "pose") drawPose();
  if (v === "emg") drawEmg();
  if (v === "camera") drawCamera();
  if (v === "model") drawModelView();
}

function showView() {
  const v = $("view").value;
  for (const name of ["pose", "emg", "camera", "model"]) $(`view-${name}`).hidden = name !== v;
  // Only pull the video while it's shown, the server stops encoding frames when nobody asks
  $("video").src = v === "camera" && config.camera ? `/video.mjpg?${Date.now()}` : "";
  $("view-note").textContent = v === "camera" && !config.camera ? "started with --no-camera" : "";
}

$("view").addEventListener("change", showView);

attachRotate($("pose"), view, () => state && drawView());
attachRotate($("target"), solidView, () => state && drawSession());
attachRotate($("model-hand"), solidView, () => state && drawView());

// Live connection

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (e) => {
    state = JSON.parse(e.data);
    stateAt = performance.now();
    // Each part on its own, so an error in one can't freeze the others (it did freeze the camera overlay once)
    for (const draw of [drawWarnings, drawSession, drawView, drawTools]) {
      try {
        draw();
      } catch (err) {
        console.error(draw.name, err);
      }
    }
  };
  ws.onclose = () => {
    $("warnings").innerHTML = '<div class="serious">Lost the connection to the recorder, retrying</div>';
    setTimeout(connect, 1000);
  };
}

async function init() {
  config = await (await fetch("/api/config")).json();
  $("source").textContent = `EMG from ${config.source}, ${config.channels.length} channels at ${config.rate} Hz`
    + (config.calibration ? ", calibration loaded" : ", no calibration");
  loadSettings();
  initTools();
  if (!config.camera) $("view").value = "pose";
  showView();
  connect();
  requestAnimationFrame(frameLoop);
}

init();
