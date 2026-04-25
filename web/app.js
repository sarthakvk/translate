/**
 * Real-time EN→HI translation client.
 *
 * Flow:
 *   1. User clicks Start → getUserMedia (16kHz mono)
 *   2. ScriptProcessor captures 4096-sample PCM chunks → send over WebSocket as base64
 *   3. Server runs VAD → ASR → translate → TTS, streams back JSON messages
 *   4. Client renders transcripts and plays MP3 audio
 *   5. Audio is queued: new phrases play after the current one finishes, so
 *      speech is never interrupted mid-sentence.
 */

const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
const SAMPLE_RATE = 16000;
const CHUNK_SAMPLES = 4096; // ~256ms at 16kHz

let ws = null;
let audioCtx = null;
let sourceNode = null;
let processorNode = null;
let audioQueue = [];         // pending base64 MP3 clips
let isPlaying = false;       // true while an Audio object is active
let currentAudio = null;     // the Audio element currently playing
let pausedForSpeech = false; // true while audio is paused waiting for user silence
let phraseStartTime = null;

const startBtn = document.getElementById("startBtn");
const stopBtn  = document.getElementById("stopBtn");
const statusBadge = document.getElementById("statusBadge");
const enBody = document.getElementById("enBody");
const hiBody = document.getElementById("hiBody");
const latencyLabel = document.getElementById("latencyLabel");

// ── Status helpers ──────────────────────────────────────────────────────────

function setStatus(state) {
  statusBadge.className = "badge badge-" + state;
  const labels = { idle: "Idle", listening: "Listening…", processing: "Processing…", speaking: "Speaking…" };
  statusBadge.textContent = labels[state] ?? state;
}

// ── Transcript rendering ────────────────────────────────────────────────────

function clearPlaceholders() {
  enBody.querySelector(".placeholder")?.remove();
  hiBody.querySelector(".placeholder")?.remove();
}

function appendPhrase(container, text, cls) {
  const el = document.createElement("p");
  el.className = "phrase " + cls;
  el.textContent = text;
  container.appendChild(el);
  container.scrollTop = container.scrollHeight;
  return el;
}

// We keep track of the last EN bubble so we can upgrade it with the Hindi pair
let lastEnEl = null;

function onTranscript(msg) {
  clearPlaceholders();
  if (msg.stage === "final" && msg.hi) {
    // Full pair: update existing EN bubble (avoid duplication) and add HI bubble
    if (lastEnEl) lastEnEl.textContent = msg.en;
    appendPhrase(hiBody, msg.hi, "phrase-hi");
    lastEnEl = null;

    // Latency
    if (phraseStartTime) {
      const ms = Date.now() - phraseStartTime;
      latencyLabel.textContent = `Last phrase: ${(ms / 1000).toFixed(1)}s end-to-end`;
      phraseStartTime = null;
    }
  } else {
    // EN-only update (transcription confirmed, translation in flight)
    lastEnEl = appendPhrase(enBody, msg.en, "phrase-en");
    setStatus("processing");
  }
}

// ── Audio playback queue ────────────────────────────────────────────────────

function playAudio(b64mp3) {
  audioQueue.push(b64mp3);
  if (!isPlaying) playNext();
}

function playNext() {
  if (audioQueue.length === 0) {
    isPlaying = false;
    currentAudio = null;
    setStatus("listening");
    return;
  }

  isPlaying = true;
  setStatus("speaking");

  const b64mp3 = audioQueue.shift();
  const blob = b64toBlob(b64mp3, "audio/mpeg");
  const url  = URL.createObjectURL(blob);
  const audio = new Audio(url);
  currentAudio = audio;

  audio.addEventListener("ended", () => {
    URL.revokeObjectURL(url);
    currentAudio = null;
    playNext();
  });

  audio.play().catch(err => {
    console.warn("Audio play failed:", err);
    currentAudio = null;
    playNext();
  });
}

function b64toBlob(b64, mime) {
  const bytes = atob(b64);
  const buf = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i);
  return new Blob([buf], { type: mime });
}

// ── WebSocket ───────────────────────────────────────────────────────────────

function openWS() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => console.log("WS connected");

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    switch (msg.type) {
      case "transcript":
        onTranscript(msg);
        break;
      case "audio":
        playAudio(msg.data);
        break;
      case "speech_start":
        if (currentAudio && !currentAudio.paused) {
          currentAudio.pause();
          pausedForSpeech = true;
          setStatus("listening");
        }
        break;
      case "speech_end":
        if (pausedForSpeech && currentAudio) {
          pausedForSpeech = false;
          currentAudio.play().catch(() => playNext());
          setStatus("speaking");
        }
        break;
      case "error":
        clearPlaceholders();
        appendPhrase(enBody, "⚠ " + msg.message, "phrase-error");
        break;
    }
  };

  ws.onerror = (e) => console.error("WS error", e);
  ws.onclose = () => {
    ws = null;
    setStatus("idle");
  };
}

// ── Mic capture ─────────────────────────────────────────────────────────────

async function startCapture() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: true, noiseSuppression: true }
  });

  audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  sourceNode = audioCtx.createMediaStreamSource(stream);

  // ScriptProcessorNode is deprecated but universally supported.
  // AudioWorklet would be the modern alternative.
  processorNode = audioCtx.createScriptProcessor(CHUNK_SAMPLES, 1, 1);

  processorNode.onaudioprocess = (e) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    const float32 = e.inputBuffer.getChannelData(0);
    // Convert float32 → int16 PCM
    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32767));
    }

    if (phraseStartTime === null) phraseStartTime = Date.now();

    const b64 = arrayBufferToBase64(int16.buffer);
    ws.send(JSON.stringify({ type: "audio", data: b64 }));
  };

  sourceNode.connect(processorNode);
  processorNode.connect(audioCtx.destination);
}

function stopCapture() {
  processorNode?.disconnect();
  sourceNode?.disconnect();
  audioCtx?.close();
  processorNode = null;
  sourceNode = null;
  audioCtx = null;
}

function arrayBufferToBase64(buffer) {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

// ── Button handlers ─────────────────────────────────────────────────────────

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  stopBtn.disabled = false;
  setStatus("listening");
  lastEnEl = null;

  openWS();
  await new Promise(r => setTimeout(r, 200)); // let WS handshake complete
  await startCapture();
});

stopBtn.addEventListener("click", () => {
  stopBtn.disabled = true;
  startBtn.disabled = false;
  setStatus("idle");

  stopCapture();
  audioQueue = [];
  isPlaying = false;
  currentAudio = null;
  pausedForSpeech = false;

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "stop" }));
    ws.close();
  }
});
