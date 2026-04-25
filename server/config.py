import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY: str = os.environ["GROQ_API_KEY"]

SAMPLE_RATE = 16000
CHANNELS = 1

# webrtcvad: 0=least aggressive, 3=most aggressive
VAD_AGGRESSIVENESS = 2
# 30ms frames required by webrtcvad at 16kHz → 480 samples
VAD_FRAME_MS = 30
VAD_FRAME_SAMPLES = SAMPLE_RATE * VAD_FRAME_MS // 1000  # 480

# How many consecutive silent frames = phrase boundary (400ms / 30ms = ~13 frames)
VAD_SILENCE_THRESHOLD_FRAMES = 13
# Minimum voiced frames before we consider it real speech (avoid single pops)
VAD_MIN_SPEECH_FRAMES = 8

WHISPER_MODEL = "whisper-large-v3"
WHISPER_LANGUAGE = "en"

TTS_VOICE = "hi-IN-SwaraNeural"  # female; alt: hi-IN-MadhurNeural (male)

FILLERS = {
    "uh", "um", "uh-huh", "uh huh", "uhh", "umm",
    "you know", "i mean", "like", "basically", "right",
    "so", "well", "actually", "literally", "honestly",
}

# Named entities that should be preserved verbatim in Hindi output.
# Extend this list freely.
NAMED_ENTITIES = [
    "Google Meet", "Google", "Zoom", "Slack", "GitHub", "GitLab",
    "Asterisk", "Jira", "Confluence", "Kubernetes", "Docker",
    "AWS", "Azure", "GCP", "API", "UI", "UX", "PR", "ETA",
]

# Formal vs casual tone detection word lists
FORMAL_MARKERS = {"please", "could", "would", "kindly", "request", "regarding", "shall"}
CASUAL_MARKERS = {"hey", "gonna", "wanna", "gotta", "yeah", "nah", "kinda", "sorta", "dunno"}
