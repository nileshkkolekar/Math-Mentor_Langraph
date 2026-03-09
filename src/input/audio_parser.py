"""Audio to text using Whisper. Returns transcript and confidence; triggers HITL if unclear."""
from pathlib import Path
from typing import Any
import numpy as np

_model = None

# Whisper expects 16 kHz mono float32. We load with soundfile when possible to avoid ffmpeg on Windows.
WHISPER_SAMPLE_RATE = 16000


def _get_whisper():
    """Lazy import openai-whisper (package name: openai-whisper, import: whisper)."""
    try:
        import whisper
        return whisper
    except Exception as e:
        raise RuntimeError(
            "Whisper not available. Install with: pip uninstall whisper; pip install openai-whisper"
        ) from e


def _get_model():
    global _model
    if _model is None:
        whisper = _get_whisper()
        _model = whisper.load_model("base")
    return _model


def _load_audio_native(path: str) -> np.ndarray | None:
    """Load audio with soundfile (no ffmpeg). Returns float32 mono at 16 kHz or None if format unsupported."""
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != WHISPER_SAMPLE_RATE:
            from scipy.signal import resample
            n = int(len(data) * WHISPER_SAMPLE_RATE / sr)
            data = resample(data, n).astype(np.float32)
        return data
    except Exception:
        return None


def _normalize_math_phrases(text: str) -> str:
    """Replace common speech phrases with math notation."""
    if not text:
        return text
    replacements = [
        ("square root of", "sqrt"),
        ("square root", "sqrt"),
        ("raised to", "^"),
        ("to the power of", "^"),
        ("divided by", "/"),
        ("times", "*"),
        ("multiplied by", "*"),
    ]
    t = text
    for a, b in replacements:
        t = t.replace(a, b)
    return t


def parse_audio(
    audio_path: str | Path | bytes,
    confidence_threshold: float = 0.6,
    filename: str | None = None,
) -> dict[str, Any]:
    """
    Transcribe audio with Whisper. Return transcript.
    If audio_path is bytes, pass filename (e.g. "audio.wav") so the temp file uses the right extension;
    WAV can be loaded without ffmpeg; MP3/M4A require ffmpeg on PATH.
    """
    if isinstance(audio_path, bytes):
        import tempfile
        suffix = Path(filename).suffix if filename else ".wav"
        if not suffix or suffix == ".":
            suffix = ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_path)
            path = f.name
        try:
            return _transcribe_path(path, confidence_threshold)
        finally:
            Path(path).unlink(missing_ok=True)
    path = str(audio_path)
    return _transcribe_path(path, confidence_threshold)


def _transcribe_path(path: str, confidence_threshold: float) -> dict[str, Any]:
    model = _get_model()
    # Try loading with soundfile first (no ffmpeg needed for WAV/FLAC/OGG)
    audio = _load_audio_native(path)
    if audio is not None:
        result = model.transcribe(audio, fp16=False)
    else:
        # Fall back to Whisper's built-in loader (requires ffmpeg on PATH for MP3 etc.)
        try:
            result = model.transcribe(path)
        except FileNotFoundError as e:
            raise RuntimeError(
                "FFmpeg not found. Whisper needs FFmpeg to load MP3/M4A and some other formats. "
                "Option 1: Install FFmpeg and add it to your PATH (see https://ffmpeg.org). "
                "Option 2: Upload a WAV file instead (WAV works without FFmpeg)."
            ) from e
    text = (result.get("text") or "").strip()
    text = _normalize_math_phrases(text)
    # Heuristic: very short or unclear markers -> low confidence
    unclear_markers = ["inaudible", "...", "[", "]", "?"]
    is_unclear = len(text) < 10 or any(m in text.lower() for m in unclear_markers)
    # Whisper doesn't give easy confidence; use 0.9 if we got substantial text else 0.5
    confidence = 0.9 if len(text) > 20 and not is_unclear else 0.5
    needs_hitl = confidence < confidence_threshold or is_unclear
    return {
        "text": text,
        "confidence": confidence,
        "needs_hitl": needs_hitl,
        "source": "audio",
    }
