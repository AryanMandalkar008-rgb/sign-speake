import { GestureRecognizer, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const video = document.querySelector("#webcam");
const canvas = document.querySelector("#overlay");
const ctx = canvas.getContext("2d");
const startBtn = document.querySelector("#startBtn");
const stopBtn = document.querySelector("#stopBtn");
const placeholder = document.querySelector("#placeholder");
const cameraStatus = document.querySelector("#cameraStatus");
const cameraNote = document.querySelector("#cameraNote");
const resultWord = document.querySelector("#resultWord");
const resultDescription = document.querySelector("#resultDescription");
const confidence = document.querySelector("#confidence");
const recognitionBar = document.querySelector("#recognitionBar");
const modelStatus = document.querySelector("#modelStatus");
const fpsLabel = document.querySelector("#fpsLabel");
const addBtn = document.querySelector("#addBtn");
const phraseOutput = document.querySelector("#phraseOutput");
const speakToggle = document.querySelector("#speakToggle");

let recognizer = null;
let stream = null;
let running = false;
let lastVideoTime = -1;
let lastResult = null;
let currentWord = "";
let phrase = [];
let lastSpoken = "";
let lastSpokenAt = 0;

const meanings = {
  Open_Palm: { word: "Hello", alt: "Stop", description: "An open palm detected. Depending on context, this may be used as a greeting or to signal stop." },
  Closed_Fist: { word: "Stop", alt: "Wait", description: "A closed fist detected. This shortcut maps the gesture to “Stop” or “Wait” for this demo." },
  Thumb_Up: { word: "Yes", alt: "Good", description: "Thumbs up detected. This demo maps it to “Yes” or “Good”." },
  Thumb_Down: { word: "No", alt: "Not good", description: "Thumbs down detected. This demo maps it to “No”." },
  Victory: { word: "Peace", alt: "Two", description: "Victory gesture detected. This demo maps it to “Peace” or “Two”." },
  Pointing_Up: { word: "Attention", alt: "One", description: "One finger pointing up detected. This demo maps it to “Attention” or “One”." },
  ILoveYou: { word: "I love you", alt: "ILY", description: "The model detected its I-love-you hand pose. Gesture meaning can vary by sign language and context." },
  Closed_Fist: { word: "Stop", alt: "Wait", description: "A closed fist detected. This demo maps the gesture to “Stop” or “Wait”." },
  Thumb_Up: { word: "Yes", alt: "Good", description: "Thumbs up detected. This demo maps it to “Yes” or “Good”." }
};

function showResult(label, score) {
  const meaning = meanings[label];
  if (!meaning) {
    currentWord = "";
    resultWord.innerHTML = 'Not sure<span>.</span>';
    resultDescription.textContent = label ? `Detected ${label.replaceAll("_", " ").toLowerCase()}, but no word shortcut is configured for it.` : "Keep one hand visible and hold a clear gesture.";
    confidence.textContent = score ? `${Math.round(score * 100)}% model confidence` : "No hand detected";
    recognitionBar.style.width = `${Math.round((score || 0) * 100)}%`;
    addBtn.disabled = true;
    return;
  }
  currentWord = meaning.word;
  resultWord.innerHTML = `${meaning.word.replaceAll("&", "&amp;")}<span>.</span>`;
  resultDescription.textContent = meaning.description;
  confidence.textContent = `${Math.round(score * 100)}% model confidence`;
  recognitionBar.style.width = `${Math.round(score * 100)}%`;
  addBtn.disabled = false;

  if (speakToggle.checked && score > 0.72) {
    const now = Date.now();
    if (meaning.word !== lastSpoken || now - lastSpokenAt > 3500) {
      say(meaning.word);
      lastSpoken = meaning.word;
      lastSpokenAt = now;
    }
  }
}

function drawLandmarks(result) {
  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h) return;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  ctx.clearRect(0, 0, w, h);
  if (!result.landmarks?.length) return;
  for (const landmarks of result.landmarks) {
    ctx.lineWidth = Math.max(2, w / 240);
    ctx.strokeStyle = "#76f2c0";
    ctx.fillStyle = "#f2fff9";
    const connections = [
      [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
      [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
      [13,17],[17,18],[18,19],[19,20],[0,17]
    ];
    for (const [a,b] of connections) {
      ctx.beginPath(); ctx.moveTo(landmarks[a].x*w, landmarks[a].y*h);
      ctx.lineTo(landmarks[b].x*w, landmarks[b].y*h); ctx.stroke();
    }
    for (const p of landmarks) {
      ctx.beginPath(); ctx.arc(p.x*w, p.y*h, Math.max(2.2,w/180), 0, Math.PI*2); ctx.fill();
    }
  }
}

async function loadModel() {
  modelStatus.textContent = "Loading model…";
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  recognizer = await GestureRecognizer.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 1
  });
  modelStatus.textContent = "Ready";
}

async function startCamera() {
  startBtn.disabled = true;
  startBtn.textContent = "Starting…";
  cameraNote.textContent = "Loading hand-gesture model…";
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera access is unavailable. Open this website using HTTPS or localhost.");
    }
    if (!recognizer) await loadModel();
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    video.srcObject = stream;
    await video.play();
    running = true;
    placeholder.style.display = "none";
    cameraStatus.classList.add("active");
    cameraStatus.innerHTML = "<i></i> Camera active";
    startBtn.textContent = "Camera running";
    stopBtn.disabled = false;
    cameraNote.textContent = "Tip: hold one hand clearly in view and keep it steady for a moment.";
    fpsLabel.textContent = "SCANNING";
    modelStatus.textContent = "Running";
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    modelStatus.textContent = "Could not start";
    cameraNote.textContent = err.name === "NotAllowedError"
      ? "Camera permission was denied. Allow camera access in your browser settings, then try again."
      : (err.message || "Could not start the camera or load the model. Check your internet connection and retry.");
    startBtn.disabled = false;
    startBtn.innerHTML = "<span>▶</span> Try again";
  }
}

function loop() {
  if (!running) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    try {
      const result = recognizer.recognizeForVideo(video, performance.now());
      drawLandmarks(result);
      const top = result.gestures?.[0]?.[0];
      if (top && top.categoryName !== "None" && top.score >= 0.45) {
        showResult(top.categoryName, top.score);
        fpsLabel.textContent = "HAND FOUND";
      } else {
        showResult("", 0);
        fpsLabel.textContent = result.landmarks?.length ? "ANALYZING" : "FINDING HAND";
      }
    } catch (err) {
      console.warn("Recognition frame skipped", err);
    }
  }
  requestAnimationFrame(loop);
}

function stopCamera() {
  running = false;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  placeholder.style.display = "flex";
  cameraStatus.classList.remove("active");
  cameraStatus.innerHTML = "<i></i> Camera off";
  startBtn.disabled = false;
  startBtn.innerHTML = "<span>▶</span> Start camera";
  stopBtn.disabled = true;
  fpsLabel.textContent = "WAITING";
  modelStatus.textContent = recognizer ? "Ready" : "Not loaded";
  cameraNote.textContent = "Camera stopped. Your video is not uploaded or saved by this website.";
  resultWord.innerHTML = 'Ready<span>.</span>';
  resultDescription.textContent = "Start the camera, then hold a clear hand gesture in view.";
  confidence.textContent = "—";
  recognitionBar.style.width = "0";
  addBtn.disabled = true;
  currentWord = "";
}

function say(text) {
  if (!("speechSynthesis" in window)) {
    cameraNote.textContent = "Text-to-speech is not supported by this browser.";
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "en-IN";
  window.speechSynthesis.speak(utterance);
}

function renderPhrase() {
  phraseOutput.textContent = phrase.length ? phrase.join(" ") : "Your recognized words will appear here…";
  phraseOutput.classList.toggle("has-text", phrase.length > 0);
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
addBtn.addEventListener("click", () => {
  if (currentWord && phrase[phrase.length - 1] !== currentWord) {
    phrase.push(currentWord);
    renderPhrase();
  }
});
document.querySelector("#clearBtn").addEventListener("click", () => { phrase = []; renderPhrase(); });
document.querySelector("#speakPhraseBtn").addEventListener("click", () => {
  if (phrase.length) say(phrase.join(" "));
});
document.querySelector("#copyBtn").addEventListener("click", async () => {
  if (!phrase.length) {
    cameraNote.textContent = "Add a recognized word to your phrase before copying.";
    return;
  }
  try {
    await navigator.clipboard.writeText(phrase.join(" "));
    cameraNote.textContent = "Phrase copied to clipboard.";
  } catch {
    cameraNote.textContent = "Clipboard access was blocked by the browser. Select and copy the phrase manually.";
  }
});
window.addEventListener("beforeunload", () => {
  if (stream) stream.getTracks().forEach(track => track.stop());
});
