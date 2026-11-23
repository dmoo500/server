"""Extract a recognition query from a bounded PCM capture."""

from __future__ import annotations

from typing import Any

import numpy as np
import soxr
from audio_fingerprint import FORMAT_ID, SAMPLE_RATE, decode_pcm, fingerprint


def extract(pcm: bytes, sample_rate: int, channels: int, encoding: str) -> dict[str, Any]:
    """Create landmarks relative to the beginning of a mono 44.1 kHz capture."""
    audio = decode_pcm(pcm, encoding, channels)
    if sample_rate != SAMPLE_RATE:
        audio = soxr.resample(audio, sample_rate, SAMPLE_RATE)
    samples = np.clip(np.asarray(audio, dtype=np.float32), -1, 1)
    landmarks = fingerprint(samples)
    if len(landmarks) > 10000:
        raise ValueError("Capture contains too many landmarks for recognition")
    return {
        "version": 1,
        "format_id": FORMAT_ID,
        "sample_rate": SAMPLE_RATE,
        "offset_unit": "fft_hops",
        "sample_count": len(samples),
        "fingerprints": landmarks,
    }
