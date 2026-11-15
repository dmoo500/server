"""Validation safeguards for the isolated fingerprint performance comparison."""

import wave
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.perf.fingerprint_streaming_benchmark import validate_evidence
from scripts.perf.perf_server import prepare_fingerprint_fixture
from scripts.perf.run_benchmark import BenchmarkError


def evidence() -> dict[str, Any]:
    """Return a complete pair of independent synthetic-track analyses."""
    return {
        "results": [
            {
                "item_id": item_id,
                "format_id": "ma-dejavu-landmarks-v2",
                "analysis_version": 2,
                "landmarks": 20000,
                "duration_seconds": 60,
                "full": True,
            }
            for item_id in ("a", "b")
        ]
    }


def test_enabled_requires_exact_track_set() -> None:
    """Every queued fixture must have its own complete result."""
    doc = evidence()
    validate_evidence(True, doc, {"a", "b"})
    with pytest.raises(BenchmarkError, match="all fixture tracks"):
        validate_evidence(True, doc, {"a", "b", "c"})
    doc["results"][1]["item_id"] = "a"
    with pytest.raises(BenchmarkError, match="all fixture tracks"):
        validate_evidence(True, doc, {"a", "b"})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("full", False),
        ("landmarks", 0),
        ("analysis_version", 1),
        ("format_id", "ma-dejavu-landmarks-v1"),
        ("duration_seconds", 59),
    ],
)
def test_rejects_invalid_analysis(field: str, value: object) -> None:
    """No partial, silent or old extraction may yield success-shaped timings."""
    doc = deepcopy(evidence())
    doc["results"][0][field] = value
    with pytest.raises(BenchmarkError, match="validation failed"):
        validate_evidence(True, doc, {"a", "b"})


def test_baseline_requires_absent_analysis() -> None:
    """An off arm that actually fingerprints is not a baseline."""
    validate_evidence(False, {"results": []}, {"a", "b"})
    with pytest.raises(BenchmarkError, match="Baseline"):
        validate_evidence(False, evidence(), {"a", "b"})


def test_fixture_is_deterministic_non_silent_pcm(tmp_path: Path) -> None:
    """Generate precisely 60 seconds of bounded stereo PCM rather than silence."""
    target = prepare_fingerprint_fixture(tmp_path)
    with wave.open(str(target), "rb") as audio:
        assert audio.getframerate() == 44100
        assert audio.getnchannels() == 2
        assert audio.getsampwidth() == 2
        assert audio.getnframes() == 60 * 44100
        first = audio.readframes(44100)
    pcm = np.frombuffer(first, dtype="<i2").reshape(-1, 2)
    assert np.array_equal(pcm[:, 0], pcm[:, 1])
    assert 2500 < np.std(pcm) < 4000
    prepare_fingerprint_fixture(tmp_path)
    with wave.open(str(target), "rb") as audio:
        assert audio.readframes(44100) == first
