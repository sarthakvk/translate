# Live EN → HI Voice Translator

Real-time voice-to-voice English → Hindi translation pipeline. Speak in English, hear it in Hindi with ~1.5–2.5 s of end-to-end latency per phrase.

---

## Pipeline Overview

```
🎙️  Browser mic (16 kHz PCM)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  VAD  —  webrtcvad, 30 ms frames                        │
│  Accumulates audio until ≥ 400 ms of silence.           │
│  Emits one phrase blob per natural pause.               │
└────────────────────────┬────────────────────────────────┘
                         │ float32 PCM array
                         ▼
┌─────────────────────────────────────────────────────────┐
│  ASR  —  Groq Whisper-large-v3 (cloud)                  │
│  PCM → WAV (in-memory) → Groq API → English text.       │
│  language="en" forced; proper-noun prompt hint.         │
└────────────────────────┬────────────────────────────────┘
                         │ raw English string
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Pre-processor  —  preprocessor.py                      │
│  1. Strip fillers: "uh", "you know", "basically", …     │
│  2. Collapse stutters: "the the meeting" → "the meeting" │
│  3. Tag named entities with placeholders (XENTX0X, …)   │
│  4. Detect tone: formal / casual / neutral              │
└────────────────────────┬────────────────────────────────┘
                         │ cleaned text + entity map + tone
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Translator  —  Google Translate (deep_translator)      │
│  Prepends last Hindi phrase as soft context prefix      │
│  to maintain discourse continuity across phrases.       │
└────────────────────────┬────────────────────────────────┘
                         │ raw Hindi string
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Post-processor  —  preprocessor.py                     │
│  Restores tagged named entities the translator          │
│  may have transliterated or mangled.                    │
└────────────────────────┬────────────────────────────────┘
                         │ clean Hindi string
                         ▼
┌─────────────────────────────────────────────────────────┐
│  TTS  —  Microsoft edge-tts (hi-IN-SwaraNeural)         │
│  Streams MP3 bytes asynchronously; no API key needed.   │
└────────────────────────┬────────────────────────────────┘
                         │ base64 MP3 over WebSocket
                         ▼
🗣️  Browser audio queue → plays clips in order
```

---

## Key Design Decisions

### Why phrase-level, not word-by-word streaming

Word-by-word translation produces grammatically broken Hindi because Hindi word order is SOV (subject–object–verb) while English is SVO. A sentence can only be correctly translated once its structure is known. The VAD buffers audio until a natural pause (≥ 400 ms of silence), then fires the full phrase through the pipeline in one shot. This trades ~0.4 s of extra latency for coherent output.

### Why Groq for ASR

Groq runs Whisper-large-v3 on dedicated LPU hardware, returning transcriptions in ~100–250 ms for a typical phrase — faster than running Whisper locally on a CPU-only or M-series Mac. No model download required; just a free API key. The alternative (`faster-whisper` locally) would take 300–600 ms on CPU and requires ~1.5 GB of disk per model.

### Audio queue, not barge-in

When you speak while Hindi audio is playing, the new phrase is processed concurrently and its audio is **queued**. Clips play back-to-back in order. Nothing is cancelled or skipped. This is more natural than stopping mid-sentence, and avoids the jarring effect of abrupt interruption.

### Named entity preservation

Google Translate often transliterates or re-translates proper nouns ("GitHub" → "गिटहब", "Zoom" → "ज़ूम"). Before translation, known entities are replaced with opaque placeholders (`XENTX0X`) that the translator passes through untouched. The post-processor substitutes the originals back.

---

## WebSocket Protocol

All messages are JSON.

**Client → Server**

| Message | When |
|---|---|
| `{"type":"audio","data":"<base64 PCM int16>"}` | Every ~256 ms chunk from MediaRecorder |
| `{"type":"stop"}` | User clicks Stop; server flushes any buffered speech |

**Server → Client**

| Message | When |
|---|---|
| `{"type":"transcript","en":"...","stage":"final"}` | ASR done, translation in flight |
| `{"type":"transcript","en":"...","hi":"...","stage":"final"}` | Translation done |
| `{"type":"audio","data":"<base64 MP3>"}` | TTS ready; client queues for playback |
| `{"type":"error","message":"..."}` | Any pipeline exception |

---

## Setup

### Docker (recommended — no local deps needed)

```bash
cp .env.example .env        # add your GROQ_API_KEY
docker compose up --build
```

Open **http://localhost:8000**.

The API key is injected at runtime via `env_file` and never baked into the image.

### Local

Requirements: Python 3.10+, a free [Groq API key](https://console.groq.com).

```bash
pip install -r requirements.txt   # Linux: may need gcc + python3-dev first
cp .env.example .env              # set GROQ_API_KEY
uvicorn server.main:app --reload
```

---

## Edge Cases

| Problem | Solution |
|---|---|
| Fillers ("uh", "you know") | Stripped by pre-processor before translation |
| Named entities ("Google Meet", "Asterisk") | Placeholder tagging → post-processor restore |
| Strong accent / fast speech | Whisper-large-v3 with forced `language="en"` |
| Partial phrase / sentence fragments | VAD holds until silence; incomplete phrases never translate |
| Stutter / repeated words | Regex collapser: "the the meeting" → "the meeting" |
| Tone (formal vs casual) | Keyword heuristic sets tone hint in translator context |
| Numbers & times ("5 PM", "ETA") | Google Translate handles most; entity list covers abbreviations |
| Speaking while TTS plays | New phrase queued; current clip finishes before next plays |

---

## Configuration

All tuneable parameters are in [`server/config.py`](server/config.py).

| Setting | Default | Effect |
|---|---|---|
| `VAD_AGGRESSIVENESS` | `2` | 0–3; higher filters more background noise |
| `VAD_SILENCE_THRESHOLD_FRAMES` | `13` | ~400 ms silence = phrase end; raise for slow speakers |
| `VAD_MIN_SPEECH_FRAMES` | `8` | Minimum voiced frames before a phrase is considered real |
| `WHISPER_MODEL` | `whisper-large-v3` | Also try `whisper-large-v3-turbo` for lower latency |
| `TTS_VOICE` | `hi-IN-SwaraNeural` | Female voice; `hi-IN-MadhurNeural` for male |
| `NAMED_ENTITIES` | list in config | Add domain-specific proper nouns here |
| `FILLERS` | set in config | Add / remove filler words |

---

## Project Structure

```
translate/
├── server/
│   ├── main.py           FastAPI app — serves UI, owns WebSocket endpoint
│   ├── pipeline.py       Async orchestrator (one instance per connection)
│   ├── vad.py            Voice Activity Detection — phrase boundary detection
│   ├── asr.py            Groq Whisper API wrapper
│   ├── preprocessor.py   Filler strip, stutter collapse, NER tagging, tone detect
│   ├── translator.py     Google Translate via deep_translator + context continuity
│   ├── tts.py            edge-tts Hindi synthesis → MP3 bytes
│   └── config.py         All tuneable parameters and word lists
├── web/
│   ├── index.html        Browser UI — mic button, EN/HI transcript panes
│   ├── app.js            WebSocket client, MediaRecorder, audio queue
│   └── style.css         Dark theme
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```
