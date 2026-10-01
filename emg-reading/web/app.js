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

function mirrored() {
  return $("setup").tracked_hand.value === "left";
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
  if (config.tracked_hand) form.tracked_hand.value = config.tracked_hand;
  updateDuration();
}

function readSettings() {
  const form = $("setup");
  const num = (name) => Number(form[name].value);
  return {
    subject: form.subject.value.trim(),
    band_arm: form.band_arm.value,
    tracked_hand: form.tracked_hand.value,
    placement: form.placement.value.trim(),
    notes: form.notes.value.trim(),
    postures: [...form.querySelectorAll("[name=posture]:checked")].map((c) => c.value),
    reps: num("reps"), hold_s: num("hold_s"), rest_s: num("rest_s"), free_s: num("free_s"),
    practice: form.practice.checked,
  };
}

// Matches gestures.session_plan, without the breaks that wait for Continue
// The server works out the length from the real session plan
async function updateDuration() {
  const s = readSettings();
  if (!s.postures.length) {
    $("duration").textContent = "No postures: only the sync taps, to check the camera delay";
    return;
  }
  try {
    const { postures, reps, hold_s, rest_s, free_s } = s;
    const plan = await post("/api/plan", { postures, reps, hold_s, rest_s, free_s });
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
  updateDuration();
  if (e.target.name === "tracked_hand") post("/api/hand", { hand: e.target.value }).catch(() => {});
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
  $("setup").hidden = !!s;
  $("cue-panel").hidden = !s;
  if (!s) {
    drawSummary();
    return;
  }
  const cue = s.cue;
  $("posture").textContent = config.postures[cue.posture] ?? cue.posture;
  $("progress-text").textContent = `cue ${s.index + 1} of ${s.count}   ${fmtTime(s.elapsed)}`;
  $("cue-text").innerHTML = s.paused ? "PAUSED<small>press Continue (space) to redo this cue</small>" : cueTitle(cue.text);
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

  const ctx = $("target").getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  const caption = $("target-label");
  caption.textContent = target ? `${target.upcoming ? "Next" : "Now"}: ${target.title}` : "";
  caption.classList.toggle("upcoming", !!target?.upcoming);
  // The next gesture is drawn darker so it doesn't read as the current one. Not transparent, the overlapping
  // parts would show through each other
  if (target) {
    drawSolidHand(ctx, target.angles, { view: solidView, mirror: mirrored(), color: target.upcoming ? UPCOMING_COLOR : SOLID_COLOR });
  }
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
    ${r.frames} camera frames, hand detected in ${detected}${r.camera_delay ? `<br>${r.camera_delay}` : ""}`;
}

// Target pose of the current cue, or the next gesture during a rest so it can be prepared
// Returns {angles, title, text, upcoming}, upcoming is true when it shows the next gesture
function targetPose() {
  const s = state.session;
  if (!s) return null;
  let cue = s.cue;
  if (cue.kind === "free" || cue.kind === "sync" || cue.kind === "break") return null;
  const upcoming = cue.label === "rest" && s.next && !!config.gestures[s.next.label];
  if (upcoming) cue = s.next;
  const g = config.gestures[cue.label];
  const [title, description] = cue.text.split("\n");
  return g ? { angles: anglesByName(g.angles), title, text: `${title}\n${description ?? ""}`.trim(), upcoming } : null;
}

function drawPose() {
  const ctx = $("pose").getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  const s = state.session;
  const cue = s?.cue;
  const g = cue && (cue.kind === "hold" || cue.kind === "finger") ? config.gestures[cue.label] : null;
  const target = g ? anglesByName(g.angles) : null;
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
}

init();
