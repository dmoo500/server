"""Tests for live fingerprint recognition."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from music_assistant_models.enums import ContentType, MediaType, ProviderFeature
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.streamdetails import StreamDetails, StreamMetadata

from music_assistant.models.media_capabilities import AudioSourceMixin
from music_assistant.providers.dejavu_fingerprint_recognize import (
    CAPTURE_LENGTHS,
    FingerprintRecognizer,
    _parse_metadata,
)
from music_assistant.providers.dejavu_fingerprint_recognize.extract import extract


def _provider(url: str) -> FingerprintRecognizer:
    mass = MagicMock()
    mass.create_task = asyncio.create_task
    mass.streams.audio_analysis.get_audio_analysis_version = AsyncMock(return_value=None)
    manifest = MagicMock(domain="dejavu_fingerprint_recognize")
    config = MagicMock(instance_id="recognizer")
    values = {
        "server_url": url,
        "api_key": "test-key",
        "application_key": "test-app-key",
        "sources": ["input|source"],
        "log_level": "GLOBAL",
    }
    config.get_value.side_effect = lambda key: values.get(key)
    mass.config.get.return_value = {
        "server_url": url,
        "api_key": "test-key",
        "application_key": "test-app-key",
    }
    mass.config.decrypt_string.side_effect = lambda value: value
    return FingerprintRecognizer(mass, manifest, config)


def _stream(item_id: str = "source") -> StreamDetails:
    return StreamDetails(
        provider="input",
        item_id=item_id,
        audio_format=AudioFormat(),
        media_type=MediaType.AUDIO_SOURCE,
        stream_metadata=StreamMetadata(title="Turntable"),
    )


def test_extract_normalizes_a_three_second_stereo_capture() -> None:
    """Extract valid relative landmarks after resampling stereo audio."""
    rng = np.random.default_rng(42)
    pcm = rng.integers(-30000, 30000, size=48000 * 3 * 2, dtype=np.int16).tobytes()
    payload = extract(pcm, 48000, 2, "pcm_s16le")
    assert payload["sample_rate"] == 44100
    assert payload["sample_count"] == 3 * 44100
    assert payload["fingerprints"]
    assert min(offset for _, offset in payload["fingerprints"]) >= 0
    assert max(offset for _, offset in payload["fingerprints"]) <= (3 * 44100 - 4096) // 2048


@pytest.mark.parametrize("status", ["no_match", "ambiguous"])
def test_unmatched_results_restore_placeholder(status: str) -> None:
    """Do not show the previous song when a new capture is inconclusive."""
    placeholder = StreamMetadata(title="Turntable")
    assert _parse_metadata(
        json.dumps({"status": status, "match": None}).encode(), placeholder, 3, 1000
    ) == (
        False,
        placeholder,
        None,
    )


def test_invalid_result_is_not_a_match() -> None:
    """Reject malformed matches."""
    with pytest.raises((TypeError, ValueError)):
        _parse_metadata(
            b'{"status":"matched","match":{"metadata":{}}}', StreamMetadata(title="x"), 3, 1000
        )


@pytest.mark.parametrize(
    ("duration", "offset", "expected_interval", "position"),
    [
        (240, 30, 60, 33),
        (240, 220, 20, 223),
        (240, 236, 5, 239),
        (None, 30, None, None),
        (240, None, None, None),
        (240, 300, None, None),
        (True, 30, None, None),
        (240, -1, None, None),
        (float("nan"), 30, None, None),
        (10**400, 30, None, None),
        (240, 10**400, None, None),
    ],
)
def test_recognition_timing(
    duration: object, offset: object, expected_interval: float | None, position: int | None
) -> None:
    """Schedule from the start of the capture and ignore unusable timing."""
    body = json.dumps(
        {
            "status": "matched",
            "match": {
                "offset_seconds": offset,
                "metadata": {"title": "Song", "artists": ["Artist"], "duration": duration},
            },
        }
    ).encode()
    matched, metadata, interval = _parse_metadata(body, StreamMetadata(title="Live"), 3, 1000)
    assert matched
    assert interval == expected_interval
    assert metadata.elapsed_time == position
    assert metadata.elapsed_time_last_updated == (1000 if position is not None else None)


@pytest.mark.parametrize(("offset", "next_interval"), [(30, 60), (220, 20)])
async def test_next_capture_uses_audio_elapsed_during_server_request(
    monkeypatch: pytest.MonkeyPatch, offset: int, next_interval: int
) -> None:
    """Wait for the song boundary or 60 seconds from the capture, including request time."""
    received = asyncio.Event()
    release = asyncio.Event()
    sizes: list[int] = []

    async def recognize(request: web.Request) -> web.Response:
        payload = await request.json()
        sizes.append(payload["sample_count"])
        if len(sizes) == 1:
            received.set()
            await release.wait()
            return web.json_response(
                {
                    "status": "matched",
                    "match": {
                        "offset_seconds": offset,
                        "metadata": {"title": "Song", "artists": ["Artist"], "duration": 240},
                    },
                }
            )
        return web.json_response({"status": "no_match", "match": None})

    app = web.Application()
    app.router.add_post("/api/v1/dejavu/recognize/landmarks", recognize)
    server = TestServer(app)
    await server.start_server()
    provider = _provider(f"http://127.0.0.1:{server.port}")
    fmt = AudioFormat(content_type=ContentType.PCM_S16LE, sample_rate=44100, channels=1)
    second = b"\0" * (44100 * 2)
    try:
        async with aiohttp.ClientSession() as client:
            cast("MagicMock", provider.mass).http_session = client
            monkeypatch.setattr(
                provider,
                "_run_offloaded",
                AsyncMock(return_value={"sample_count": 132300, "fingerprints": [("a" * 20, 0)]}),
            )
            monkeypatch.setattr(provider, "report_stream_metadata", MagicMock(return_value=True))
            await provider.handle_async_init()
            assert await provider._start_analysis("session", _stream(), fmt)
            session = provider._live["session"]
            for _ in range(3):
                await provider.process_pcm_chunk("session", second)
            await received.wait()
            for _ in range(10):
                await provider.process_pcm_chunk("session", second)
            release.set()
            assert session.task is not None
            await session.task
            assert session.interval == next_interval
            provider._next_request = 0
            for _ in range(next_interval - 11):
                await provider.process_pcm_chunk("session", second)
            assert len(sizes) == 1
            await provider.process_pcm_chunk("session", second)
            assert session.task is not None
            await session.task
            assert len(sizes) == 2
            await provider.cancel("session")
    finally:
        release.set()
        await server.close()


async def test_source_options_include_sendspin_style_player_provider() -> None:
    """Offer sources from any provider with the live source capability."""
    provider = _provider("http://127.0.0.1")
    source_provider = MagicMock(spec=AudioSourceMixin)
    source_provider.instance_id = "sendspin"
    source_provider.name = "Sendspin"
    source_provider.supported_features = {ProviderFeature.AUDIO_SOURCE}
    source_provider.get_audio_sources = AsyncMock(
        return_value=[MagicMock(item_id="line-in", name="Line-in")]
    )
    cast("MagicMock", provider.mass).providers = [source_provider]
    entry = (await provider.get_config_entries())[0]
    assert entry.options is not None
    assert entry.options[0].value == "sendspin|line-in"


async def test_live_recognition_grows_captures_after_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submit progressively longer captures and publish a successful result."""
    sizes: list[int] = []
    responses = ["no_match", "ambiguous", "no_match", "matched"]

    async def recognize(request: web.Request) -> web.Response:
        assert request.headers["X-API-Key"] == "test-key"
        assert request.headers["X-Application-Key"] == "test-app-key"
        payload = await request.json()
        sizes.append(payload["sample_count"])
        assert payload["fingerprints"]
        status = responses[len(sizes) - 1]
        match = (
            {"metadata": {"title": "Song", "artists": ["Artist"], "album": "Album"}}
            if status == "matched"
            else None
        )
        return web.json_response({"status": status, "match": match})

    app = web.Application()
    app.router.add_post("/api/v1/dejavu/recognize/landmarks", recognize)
    server = TestServer(app)
    await server.start_server()
    port = server.port
    provider = _provider(f"http://127.0.0.1:{port}")
    fmt = AudioFormat(content_type=ContentType.PCM_S16LE, sample_rate=44100, channels=1)
    rng = np.random.default_rng(39)
    second = rng.integers(-30000, 30000, size=44100, dtype=np.int16).tobytes()
    try:
        async with aiohttp.ClientSession() as client:
            cast("MagicMock", provider.mass).http_session = client
            monkeypatch.setattr(provider, "report_stream_metadata", MagicMock(return_value=True))
            monkeypatch.setattr(
                provider, "_run_offloaded", AsyncMock(side_effect=asyncio.to_thread)
            )
            await provider.handle_async_init()
            assert not await provider._start_analysis("other", _stream("other"), fmt)
            assert await provider._start_analysis("session", _stream(), fmt)
            session = provider._live["session"]
            for attempt, seconds in enumerate((3, 5, 5, 5)):
                for _ in range(seconds):
                    await provider.process_pcm_chunk("session", second)
                assert session.task is not None
                await session.task
                assert sizes == [length * 44100 for length in CAPTURE_LENGTHS[: attempt + 1]]
                provider._next_request = 0
            assert session.capture_index == 0
            assert session.interval == 30
            last_report = cast("MagicMock", provider.report_stream_metadata).call_args.args[1]
            assert (last_report.title, last_report.artist, last_report.album) == (
                "Song",
                "Artist",
                "Album",
            )
            assert len(session.buffer) <= 15 * 44100 * 2
            await provider.cancel("session")
            assert "session" not in provider._live
    finally:
        await server.close()


async def test_retry_after_stops_requests_without_losing_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Respect server rate limits across future input chunks."""
    calls = 0

    async def busy(_request: web.Request) -> web.Response:
        nonlocal calls
        calls += 1
        return web.Response(status=429, headers={"Retry-After": "120"})

    app = web.Application()
    app.router.add_post("/api/v1/dejavu/recognize/landmarks", busy)
    server = TestServer(app)
    await server.start_server()
    port = server.port
    provider = _provider(f"http://127.0.0.1:{port}")
    fmt = AudioFormat(content_type=ContentType.PCM_S16LE, sample_rate=44100, channels=1)
    try:
        async with aiohttp.ClientSession() as client:
            cast("MagicMock", provider.mass).http_session = client
            monkeypatch.setattr(
                provider,
                "_run_offloaded",
                AsyncMock(return_value={"sample_count": 132300, "fingerprints": [("a" * 20, 0)]}),
            )
            await provider.handle_async_init()
            assert await provider._start_analysis("session", _stream(), fmt)
            session = provider._live["session"]
            for _ in range(3):
                await provider.process_pcm_chunk("session", b"\0" * (44100 * 2))
            assert session.task is not None
            await session.task
            assert provider._next_request - asyncio.get_running_loop().time() > 115
            for _ in range(10):
                await provider.process_pcm_chunk("session", b"\0" * (44100 * 2))
            assert calls == 1
            await provider.cancel("session")
    finally:
        await server.close()


async def test_server_error_retries_with_shortest_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the backoff but avoid repeatedly submitting a longer refused capture."""
    sizes: list[int] = []

    async def recognize(request: web.Request) -> web.Response:
        payload = await request.json()
        sizes.append(payload["sample_count"])
        if len(sizes) == 2:
            return web.Response(status=503)
        return web.json_response({"status": "no_match", "match": None})

    async def extract_capture(
        _function: Any, pcm: bytes, _sample_rate: int, _channels: int, _encoding: str
    ) -> dict[str, Any]:
        return {
            "sample_count": len(pcm) // 2,
            "fingerprints": [("a" * 20, 0)],
        }

    app = web.Application()
    app.router.add_post("/api/v1/dejavu/recognize/landmarks", recognize)
    server = TestServer(app)
    await server.start_server()
    provider = _provider(f"http://127.0.0.1:{server.port}")
    fmt = AudioFormat(content_type=ContentType.PCM_S16LE, sample_rate=44100, channels=1)
    try:
        async with aiohttp.ClientSession() as client:
            cast("MagicMock", provider.mass).http_session = client
            monkeypatch.setattr(provider, "_run_offloaded", AsyncMock(side_effect=extract_capture))
            monkeypatch.setattr(provider, "report_stream_metadata", MagicMock(return_value=True))
            await provider.handle_async_init()
            assert await provider._start_analysis("session", _stream(), fmt)
            session = provider._live["session"]
            for seconds in (3, 5):
                for _ in range(seconds):
                    await provider.process_pcm_chunk("session", b"\0" * (44100 * 2))
                assert session.task is not None
                await session.task
                if seconds == 3:
                    provider._next_request = 0
            assert sizes == [3 * 44100, 5 * 44100]
            assert session.capture_index == 0
            assert provider._next_request - asyncio.get_running_loop().time() > 55
            for _ in range(5):
                await provider.process_pcm_chunk("session", b"\0" * (44100 * 2))
            assert len(sizes) == 2
            provider._next_request = 0
            await provider.process_pcm_chunk("session", b"\0" * (44100 * 2))
            assert session.task is not None
            await session.task
            assert sizes == [3 * 44100, 5 * 44100, 3 * 44100]
            await provider.cancel("session")
    finally:
        await server.close()
