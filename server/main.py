"""
FastAPI application.

Routes:
  GET  /          → serves web/index.html
  GET  /static/*  → serves web/ assets
  WS   /ws        → real-time translation WebSocket

WebSocket message protocol:
  Client → Server (JSON):
    {"type": "audio", "data": "<base64 raw PCM int16 at 16kHz>"}
    {"type": "tts_done"}    ← client signals TTS playback finished
    {"type": "stop"}        ← client signals end of session / flush

  Server → Client (JSON):
    {"type": "transcript", "en": str, "stage": "final"}
    {"type": "transcript", "en": str, "hi": str, "stage": "final"}
    {"type": "audio",      "data": "<base64 MP3>"}
    {"type": "cancel"}
    {"type": "error",      "message": str}
"""

import base64
import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EN→HI Live Translator")

WEB_DIR = Path(__file__).parent.parent / "web"
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    pipeline = Pipeline()
    logger.info("Client connected")

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            kind = msg.get("type")

            if kind == "audio":
                pcm_bytes = base64.b64decode(msg["data"])
                async for out in pipeline.feed(pcm_bytes):
                    await websocket.send_text(json.dumps(out))

            elif kind == "tts_done":
                pipeline.tts_done()

            elif kind == "stop":
                async for out in pipeline.stop():
                    await websocket.send_text(json.dumps(out))
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as exc:
        logger.exception("Unexpected error")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
