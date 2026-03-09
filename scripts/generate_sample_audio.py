"""Generate sample WAV files for testing the Audio upload in Math Mentor. Run from repo root: python scripts/generate_sample_audio.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "sample_audio"
OUT_DIR.mkdir(exist_ok=True)

# Use standard library only so script runs without extra deps
import wave
import struct
import math

SAMPLE_RATE = 16000  # 16 kHz, same as Whisper's expected input


def write_wav(path: Path, samples: list[float], rate: int = SAMPLE_RATE):
    """Write float samples (-1 to 1) to a 16-bit mono WAV file."""
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        for s in samples:
            x = max(-1, min(1, s))
            wav.writeframes(struct.pack("<h", int(x * 32767)))


def main():
    # 1. Silence (2 seconds) – tests that upload and pipeline run
    n_silence = 2 * SAMPLE_RATE
    write_wav(OUT_DIR / "sample_silence.wav", [0.0] * n_silence)
    print(f"Created {OUT_DIR / 'sample_silence.wav'} (2 s silence)")

    # 2. Tone (2 seconds, 440 Hz) – tests non-speech; Whisper may return empty or "music"
    duration = 2
    n_tone = duration * SAMPLE_RATE
    freq = 440
    tone = [0.3 * math.sin(2 * math.pi * freq * i / SAMPLE_RATE) for i in range(n_tone)]
    write_wav(OUT_DIR / "sample_tone.wav", tone)
    print(f"Created {OUT_DIR / 'sample_tone.wav'} (2 s, 440 Hz tone)")

    # 3. Short "speech-like" burst (no real words – for testing without recording)
    # Beeps that might transcribe as something; or just leave as minimal
    n_beep = SAMPLE_RATE // 2  # 0.5 s
    beep = [0.5 * math.sin(2 * math.pi * 880 * i / SAMPLE_RATE) if i < n_beep // 2 else 0.0 for i in range(n_beep)]
    write_wav(OUT_DIR / "sample_beep.wav", beep)
    print(f"Created {OUT_DIR / 'sample_beep.wav'} (0.5 s beep)")

    print(f"\nSamples are in: {OUT_DIR.absolute()}")
    print("Use these in the app: Audio mode -> Upload audio -> choose a .wav file.")
    print("For real math questions: record yourself (e.g. phone voice memo) saying a problem and save as WAV or MP3.")


if __name__ == "__main__":
    main()
