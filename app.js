import { GestureRecognizer, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const $ = (s) => document.querySelector(s);
const video = $("#webcam"), canvas = $("#overlay"), ctx = canvas.getContext("2d");
const startBtn = $("#startBtn"), stopBtn = $("#stopBtn"), placeholder = $("#placeholder");
const cameraStatus = $("#cameraStatus"), cameraNote = $("#cameraNote");
const resultWord = $("#resultWord"), resultDescription = $("#resultDescription");
const confidence = $("#confidence"), recognitionBar = $("#recognitionBar");
const modelStatus = $("#modelStatus"), fpsLabel = $("#fpsLabel"), addBtn = $("#addBtn");
const phraseOutput = $("#phraseOutput"), speakToggle = $("#speakToggle");
const customLabel = $("#customLabel"), captureSampleBtn = $("#captureSampleBtn");
const clearTrainingBtn = $("#clearTrainingBtn"), trainingStatus = $("#trainingStatus");
const customList = $("#customList"), customCount = $("#customCount");

let recognizer = null, stream = null, running = false, loading = false;
let lastVideoTime = -1, currentWord = "", phrase = [], lastSpoken = "", lastSpokenAt = 0;
let latestLandmarks = null;
const STORAGE_KEY = "signspeak_custom_gestures_v1";
let customSamples = loadSamples();

const meanings = {
  Open_Palm: { word: "Hello", description: "Open palm detected. This is a general gesture shortcut, not a formal sign translation." },
  Closed_Fist: { word: "Stop", description: "Closed fist detected. This demo maps it to “Stop” or “Wait”." },
  Thumb_Up: { word: "Yes", description: "Thumbs up detected. This demo maps it to “Yes” or “Good”." },
  Thumb_Down: { word: "No", description: "Thumbs down detected. This demo maps it to “No”." },
  Victory: { word: "Peace", description: "Victory gesture detected. This demo maps it to “Peace” or “Two”." },
  Pointing_Up: { word: "Attention", description: "One finger pointing up detected. This demo maps it to “Attention” or “One”." },
  ILoveYou: { word: "I love you", description: "The model detected its I-love-you hand pose. Meaning varies by sign language and context." },
  Pointing_Down: { word: "Point down", description: "Pointing-down gesture detected." },
  Open_Palm: { word: "Hello", description: "Open palm detected. This is a general gesture shortcut, not a formal sign translation." }
};

function loadSamples() {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    return Array.isArray(value) ? value.filter(x => x && x.label && Array.isArray(x.vector)) : [];
  } catch { return []; }
}
function saveSamples() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(customSamples));
    renderCustomList();
  } catch {
    trainingStatus.textContent = "Could not save training examples. Your browser may have disabled local storage.";
  }
}
function renderCustomList() {
  customList.replaceChildren();
  const counts = {};
  customSamples.forEach(s => counts[s.label] = (counts[s.label] || 0) + 1);
  const labels = Object.keys(counts);
  customCount.textContent = `${customSamples.length} example${customSamples.length === 1 ? "" : "s"} · ${labels.length} gesture${labels.length === 1 ? "" : "s"}`;
  if (!labels.length) {
    const empty = document.createElement("span"); empty.className = "empty-custom"; empty.textContent = "No custom gestures trained yet."; customList.append(empty); return;
  }
  labels.forEach(label => {
    const chip = document.createElement("div"); chip.className = "custom-chip";
    const title = document.createElement("span"); title.textContent = `${label} (${counts[label]})`;
    const del = document.createElement("button"); del.type = "button"; del.title = `Delete ${label}`; del.textContent = "×";
    del.addEventListener("click", () => { customSamples = customSamples.filter(s => s.label !== label); saveSamples(); trainingStatus.textContent = `Removed custom gesture “${label}”.`; });
    chip.append(title, del); customList.append(chip);
  });
}
function setNote(message) { cameraNote.textContent = message; }

function normalizeLandmarks(points) {
  if (!points || points.length !== 21) return null;
  const origin = points[0];
  const centered = points.map(p => [p.x - origin.x, p.y - origin.y, (p.z || 0) - (origin.z || 0)]);
  let scale = 0;
  for (const p of centered) scale = Math.max(scale, Math.hypot(p[0], p[1], p[2]));
  if (!scale) return null;
  return centered.flatMap(p => p.map(v => v / scale));
}
function vectorDistance(a, b) {
  if (!a || !b || a.length !== b.length) return Infinity;
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return Math.sqrt(sum / a.length);
}
function classifyCustom(points) {
  const vector = normalizeLandmarks(points);
  if (!vector || !customSamples.length) return null;
  const bestByLabel = {};
  for (const sample of customSamples) {
    const distance = vectorDistance(vector, sample.vector);
    if (distance < (bestByLabel[sample.label] ?? Infinity)) bestByLabel[sample.label] = distance;
  }
  const ranked = Object.entries(bestByLabel).sort((a,b) => a[1] - b[1]);
  if (!ranked.length || ranked[0][1] > 0.12) return null;
  return { label: ranked[0][0], score: Math.max(0.5, Math.min(0.99, 1 - ranked[0][1] * 3.2)), distance: ranked[0][1] };
}

function showResult(label, score, custom = false) {
  const meaning = custom ? { word: label, description: `Matched your custom trained hand pose (${Math.round(score * 100)}% similarity estimate). This is a local example match, not a verified sign-language translation.` } : meanings[label];
  if (!meaning) {
    currentWord = "";
    resultWord.textContent = "Not sure";
    const dot = document.createElement("span"); dot.textContent = "."; resultWord.append(dot);
    resultDescription.textContent = label ? `Detected ${label.replaceAll("_", " ").toLowerCase()}, but no word shortcut is configured.` : "Keep one hand visible and hold a clear gesture in view.";
    confidence.textContent = label ? `${Math.round(score * 100)}% model confidence` : "No hand detected";
    recognitionBar.style.width = `${Math.round((score || 0) * 100)}%`; addBtn.disabled = true; return;
  }
  currentWord = meaning.word;
  resultWord.textContent = meaning.word;
  const dot = document.createElement("span"); dot.textContent = "."; resultWord.append(dot);
  resultDescription.textContent = meaning.description;
  confidence.textContent = `${Math.round(score * 100)}% ${custom ? "similarity" : "model confidence"}`;
  recognitionBar.style.width = `${Math.round(score * 100)}%`; addBtn.disabled = false;
  if (speakToggle.checked && score > 0.72) {
    const now = Date.now();
    if (meaning.word !== lastSpoken || now - lastSpokenAt > 3500) {
      say(meaning.word); lastSpoken = meaning.word; lastSpokenAt = now;
    }
  }
}
function drawLandmarks(result) {
  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h) return;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  ctx.clearRect(0, 0, w, h);
  if (!result.landmarks?.length) return;
  const connections = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
  for (const landmarks of result.landmarks) {
    ctx.lineWidth = Math.max(2, w / 240); ctx.strokeStyle = "#76f2c0"; ctx.fillStyle = "#f2fff9";
    for (const [a,b] of connections) { ctx.beginPath(); ctx.moveTo(landmarks[a].x*w,landmarks[a].y*h); ctx.lineTo(landmarks[b].x*w,landmarks[b].y*h); ctx.stroke(); }
    for (const p of landmarks) { ctx.beginPath(); ctx.arc(p.x*w,p.y*h,Math.max(2.2,w/180),0,Math.PI*2); ctx.fill(); }
  }
}
async function loadModel() {
  if (recognizer) return;
  modelStatus.textContent = "Loading model…";
  setNote("Loading the hand recognition model. Keep this page open and connected to the internet.");
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  const modelAssetPath = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";
  try {
    recognizer = await GestureRecognizer.createFromOptions(vision, { baseOptions: { modelAssetPath, delegate: "GPU" }, runningMode: "VIDEO", numHands: 1 });
  } catch (e) {
    console.warn("GPU setup failed; retrying CPU", e);
    recognizer = await GestureRecognizer.createFromOptions(vision, { baseOptions: { modelAssetPath, delegate: "CPU" }, runningMode: "VIDEO", numHands: 1 });
  }
  modelStatus.textContent = "Ready";
}
async function startCamera() {
  if (loading || running) return;
  loading = true; startBtn.disabled = true; startBtn.textContent = "Starting…"; modelStatus.textContent = "Checking camera…";
  try {
    if (!window.isSecureContext) throw new Error("Camera access requires HTTPS. Open the GitHub Pages https:// website, not a local file.");
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access is unavailable in this browser. Try current Chrome or Edge.");
    setNote("Requesting camera permission…");
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } } });
    video.srcObject = stream;
    await new Promise((resolve,reject) => { if (video.readyState >= 2) return resolve(); video.onloadedmetadata = resolve; video.onerror = () => reject(new Error("Camera preview could not be opened.")); });
    await video.play(); await loadModel();
    running = true; placeholder.style.display = "none"; cameraStatus.classList.add("active"); cameraStatus.innerHTML = "<i></i> Camera active";
    startBtn.textContent = "Camera running"; stopBtn.disabled = false;
    setNote("Camera active. Hold one hand clearly in view. Custom examples are stored only in this browser.");
    fpsLabel.textContent = "SCANNING"; modelStatus.textContent = "Running"; captureSampleBtn.disabled = false;
    requestAnimationFrame(loop);
  } catch (err) {
    console.error("SignSpeak start error:", err);
    if (stream) stream.getTracks().forEach(t => t.stop()); stream = null; video.srcObject = null; running = false;
    cameraStatus.classList.remove("active"); cameraStatus.innerHTML = "<i></i> Camera off"; modelStatus.textContent = "Start failed";
    let message = err?.message || "Could not start the camera.";
    if (err?.name === "NotAllowedError" || err?.name === "PermissionDeniedError") message = "Camera permission blocked. Allow camera access in browser site settings, then reload.";
    else if (err?.name === "NotFoundError") message = "No camera found. Connect or enable a webcam.";
    else if (err?.name === "NotReadableError") message = "Camera may be in use by another app. Close other camera apps and retry.";
    setNote(message + " If model loading fails, check internet access.");
    startBtn.disabled = false; startBtn.innerHTML = "<span>▶</span> Try again"; stopBtn.disabled = true; captureSampleBtn.disabled = true;
  } finally { loading = false; }
}
function loop() {
  if (!running) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    try {
      const result = recognizer.recognizeForVideo(video, performance.now());
      drawLandmarks(result);
      latestLandmarks = result.landmarks?.[0] || null;
      const custom = latestLandmarks ? classifyCustom(latestLandmarks) : null;
      const top = result.gestures?.[0]?.[0];
      if (custom && (!top || top.categoryName === "None" || custom.distance < 0.075)) {
        showResult(custom.label, custom.score, true); fpsLabel.textContent = "CUSTOM MATCH";
      } else if (top && top.categoryName !== "None" && top.score >= 0.45) {
        showResult(top.categoryName, top.score); fpsLabel.textContent = "HAND FOUND";
      } else { showResult("", 0); fpsLabel.textContent = latestLandmarks ? "ANALYZING" : "FINDING HAND"; }
    } catch (err) { console.warn("Recognition frame skipped", err); }
  }
  requestAnimationFrame(loop);
}
function stopCamera() {
  running = false; if (stream) stream.getTracks().forEach(t => t.stop()); stream = null; video.srcObject = null;
  ctx.clearRect(0,0,canvas.width,canvas.height); latestLandmarks = null; placeholder.style.display = "flex";
  cameraStatus.classList.remove("active"); cameraStatus.innerHTML = "<i></i> Camera off";
  startBtn.disabled = false; startBtn.innerHTML = "<span>▶</span> Start camera"; stopBtn.disabled = true;
  fpsLabel.textContent = "WAITING"; modelStatus.textContent = recognizer ? "Ready" : "Not loaded";
  setNote("Camera stopped. Your video is not uploaded or saved."); resultWord.textContent = "Ready"; const dot = document.createElement("span"); dot.textContent = "."; resultWord.append(dot);
  resultDescription.textContent = "Start the camera, then hold a clear hand gesture in view."; confidence.textContent = "—"; recognitionBar.style.width = "0"; addBtn.disabled = true; captureSampleBtn.disabled = true; currentWord = "";
}
function say(text) {
  if (!("speechSynthesis" in window)) { setNote("Text-to-speech is not supported by this browser."); return; }
  window.speechSynthesis.cancel(); const utterance = new SpeechSynthesisUtterance(text); utterance.lang = "en-IN"; window.speechSynthesis.speak(utterance);
}
function renderPhrase() {
  phraseOutput.textContent = phrase.length ? phrase.join(" ") : "Your recognized words will appear here…";
  phraseOutput.classList.toggle("has-text", phrase.length > 0);
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
addBtn.addEventListener("click", () => { if (currentWord && phrase[phrase.length - 1] !== currentWord) { phrase.push(currentWord); renderPhrase(); } });
$("#clearBtn").addEventListener("click", () => { phrase = []; renderPhrase(); });
$("#speakPhraseBtn").addEventListener("click", () => { if (phrase.length) say(phrase.join(" ")); });
$("#copyBtn").addEventListener("click", async () => {
  if (!phrase.length) { setNote("Add a recognized word to your phrase before copying."); return; }
  try { await navigator.clipboard.writeText(phrase.join(" ")); setNote("Phrase copied to clipboard."); }
  catch { setNote("Clipboard blocked. Select and copy the phrase manually."); }
});
captureSampleBtn.addEventListener("click", () => {
  const label = customLabel.value.trim();
  if (!label) { trainingStatus.textContent = "Enter a word or gesture name first."; customLabel.focus(); return; }
  if (!latestLandmarks) { trainingStatus.textContent = "No hand found yet. Put your hand in the camera frame first."; return; }
  const vector = normalizeLandmarks(latestLandmarks);
  if (!vector) { trainingStatus.textContent = "Could not read this hand pose. Try again with your hand fully visible."; return; }
  customSamples.push({ label, vector, createdAt: Date.now() });
  saveSamples();
  const n = customSamples.filter(s => s.label === label).length;
  trainingStatus.textContent = `Saved example ${n} for “${label}”. Keep the same pose and save 5–10 examples from slightly different positions.`;
});
clearTrainingBtn.addEventListener("click", () => {
  if (!confirm("Delete all custom gesture examples saved in this browser?")) return;
  customSamples = []; saveSamples(); trainingStatus.textContent = "All custom training examples cleared.";
});
customLabel.addEventListener("keydown", e => { if (e.key === "Enter") captureSampleBtn.click(); });
window.addEventListener("beforeunload", () => { if (stream) stream.getTracks().forEach(t => t.stop()); });
renderCustomList();
