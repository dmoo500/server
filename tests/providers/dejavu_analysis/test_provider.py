"""Fingerprint analysis lifecycle, consent, and upload recovery tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from aiohttp import ClientSession, web
from music_assistant_fingerprint.codec import ENCODING_ID, decode_document
from music_assistant_models.enums import ConfigEntryType, ContentType, ExternalID, MediaType
from music_assistant_models.errors import InvalidDataError
from music_assistant_models.media_items import Album, Artist, AudioFormat, Track
from music_assistant_models.queue_item import QueueItem
from music_assistant_models.streamdetails import StreamDetails
from music_assistant_models.unique_list import UniqueList

from music_assistant.controllers.streams.audio_analysis_codec import encode
from music_assistant.controllers.streams.audio_analysis_database import (
    AudioAnalysisDatabaseMixin,
    create_analysis_tables,
)
from music_assistant.controllers.streams.constants import (
    AA_DB_FILENAME,
    AA_TABLE_ANALYSIS,
    AA_TABLE_SETTINGS,
)
from music_assistant.helpers.app_vars import app_var
from music_assistant.helpers.database import DatabaseConnection
from music_assistant.models.audio_analysis import AudioAnalysisData, AudioAnalysisError
from music_assistant.providers.dejavu_analysis import (
    CONF_APPLICATION_KEY,
    CONF_KEY,
    CONF_UPLOAD,
    CONF_URL,
    MAX_RESULT_BYTES,
    MAX_RESULT_HASHES,
    DejavuAnalysisProvider,
)


@pytest.fixture(autouse=True)
def _isolated_application_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never resolve developer credentials during provider tests."""
    monkeypatch.setattr("music_assistant.providers.dejavu_analysis.app_var", lambda _name: "")


async def _provider(
    upload: bool = False, application_key: str | None = None
) -> DejavuAnalysisProvider:
    mass = MagicMock()
    mass.streams.audio_analysis.get_audio_analysis_version = AsyncMock(return_value=None)
    mass.streams.audio_analysis.set_audio_analysis = AsyncMock()
    mass.music.tracks.get_library_item_by_prov_id = AsyncMock(return_value=None)
    mass.cache.get = AsyncMock(return_value=None)
    mass.cache.set = AsyncMock()
    config = MagicMock()
    config.instance_id = "dejavu_analysis_test"
    values = {
        CONF_UPLOAD: upload,
        CONF_URL: "http://localhost:8097",
        CONF_KEY: "test-key",
        CONF_APPLICATION_KEY: application_key,
    }
    config.get_value.side_effect = lambda key, *_args: values.get(key, "GLOBAL")
    manifest = MagicMock(domain="dejavu_analysis")
    provider = DejavuAnalysisProvider(mass, manifest, config)
    provider._run_offloaded = AsyncMock(  # type: ignore[method-assign]
        side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)
    )
    await provider.handle_async_init()
    response = MagicMock(status=200)
    raw = json.dumps({"api_version": 1, "fingerprint_encodings": [ENCODING_ID]}).encode()
    response.content.readexactly = AsyncMock(side_effect=asyncio.IncompleteReadError(raw, 4097))
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=response)
    context.__aexit__ = AsyncMock(return_value=False)
    mass.http_session.get.return_value = context
    mass.streams.audio_analysis.migrate_provider_headers = AsyncMock(return_value=True)
    return provider


def _stream() -> StreamDetails:
    return StreamDetails(
        provider="test",
        item_id="track",
        duration=5,
        audio_format=AudioFormat(
            content_type=ContentType.PCM_F32LE, sample_rate=44100, bit_depth=32, channels=1
        ),
        path="https://private-provider.example/audio?token=secret",
    )


async def _analyze(provider: DejavuAnalysisProvider) -> StreamDetails:
    stream = _stream()
    assert await provider.start_analysis("session", stream, stream.audio_format)
    samples = np.random.default_rng(1).normal(0, 0.1, 44100 * 5).astype(np.float32)
    for part in np.array_split(samples, 5):
        await provider.process_pcm_chunk("session", part.tobytes())
    return stream


async def test_analysis_persists_without_upload_or_metadata_fetch() -> None:
    """Private analysis persists useful fingerprints without network requests."""
    provider = await _provider()
    assert isinstance(provider.mass, MagicMock)
    await _analyze(provider)

    await provider.finalize("session")

    saved = provider.mass.streams.audio_analysis.set_audio_analysis.call_args.kwargs["analysis"]
    assert saved.extra_data["dejavu"]["fingerprints"]
    assert saved.extra_data["upload_destination"] is None
    assert "private-provider" not in json.dumps(saved.to_dict())
    provider.mass.http_session.post.assert_not_called()
    provider.mass.music.tracks.get_library_item_by_prov_id.assert_not_awaited()
    assert not provider._data
    assert not provider._sessions


async def test_cancel_discards_fingerprints_without_upload() -> None:
    """Cancellation cannot persist or submit fingerprint results."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    await _analyze(provider)
    await provider.cancel("session")
    await provider.finalize("session")
    provider.mass.streams.audio_analysis.set_audio_analysis.assert_not_awaited()
    provider.mass.http_session.post.assert_not_called()
    assert not provider._data


async def test_shared_metadata_uses_existing_queue_track_without_provider_fetch() -> None:
    """Tracks outside the library retain queued metadata without another network lookup."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    stream = _stream()
    stream.queue_id = "queue"
    track = Track(
        item_id=stream.item_id,
        provider=stream.provider,
        name="Queued track",
        artists=UniqueList(
            [
                Artist(
                    item_id="artist", provider="test", name="Queued artist", provider_mappings=set()
                )
            ]
        ),
        album=Album(item_id="album", provider="test", name="Queued album", provider_mappings=set()),
        provider_mappings=set(),
        external_ids={(ExternalID.ISRC, "USAAA2600001")},
    )
    item = QueueItem.from_media_item("queue", track)
    item.streamdetails = stream
    provider.mass.player_queues.items.side_effect = lambda *_args: [item]
    assert await provider.start_analysis("session", stream, stream.audio_format)
    metadata = provider._data["session"].metadata
    assert metadata["title"] == "Queued track"
    assert metadata["artists"] == ["Queued artist"]
    assert metadata["album"] == "Queued album"
    assert metadata["isrc"] == "USAAA2600001"
    provider.mass.music.get_item_by_uri.assert_not_called()


async def test_queue_fallback_does_not_use_a_different_track() -> None:
    """Another queue item's metadata must never be attached to the analyzed recording."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    stream = _stream()
    stream.queue_id = "queue"
    other = MagicMock()
    other.streamdetails.uri = "track://other/track"
    provider.mass.player_queues.items.side_effect = [[other], []]
    assert await provider.start_analysis("session", stream, stream.audio_format)
    assert provider._data["session"].metadata == {"duration": 5}


async def test_incomplete_window_cannot_be_finalized() -> None:
    """Unfinished planned windows are not accepted as complete analysis."""
    provider = await _provider()
    stream = _stream()
    await provider.start_analysis("session", stream, stream.audio_format)
    with pytest.raises(AudioAnalysisError, match="track end"):
        await provider._finalize("session")


async def test_storage_limit_rejects_an_oversized_document() -> None:
    """The actual serialized document must stay below the byte storage limit."""
    provider = await _provider()
    await _analyze(provider)
    provider._data["session"].metadata["title"] = "x" * MAX_RESULT_BYTES
    with pytest.raises(AudioAnalysisError, match="storage limit"):
        await provider._finalize("session")


async def test_hash_limit_rejects_excessive_landmarks() -> None:
    """Fingerprint counts are bounded before serialization or persistence."""
    provider = await _provider()
    await _analyze(provider)
    provider._data["session"].hashes = [("0" * 20, 0)] * (MAX_RESULT_HASHES + 1)
    with pytest.raises(AudioAnalysisError, match="count"):
        await provider._finalize("session")


async def test_nonzero_seek_is_not_analyzed() -> None:
    """Seeking must not create a reference with incorrect absolute offsets."""
    provider = await _provider()
    stream = _stream()
    stream.seek_position = 1
    assert not await provider.start_analysis("session", stream, stream.audio_format)


async def test_radio_is_not_analyzed() -> None:
    """Unbounded radio streams are not song references."""
    provider = await _provider()
    stream = _stream()
    stream.media_type = MediaType.RADIO
    assert not await provider.start_analysis("session", stream, stream.audio_format)


async def _envelope(provider: DejavuAnalysisProvider) -> dict[str, Any]:
    await _analyze(provider)
    analysis = await provider._finalize("session")
    assert analysis is not None
    assert analysis.extra_data is not None
    return analysis.extra_data


def _response(provider: DejavuAnalysisProvider, status: int, document: dict[str, Any]) -> None:
    assert isinstance(provider.mass, MagicMock)
    response = MagicMock(status=status, headers={})
    response.content.readexactly = AsyncMock(
        side_effect=asyncio.IncompleteReadError(json.dumps(document).encode(), 4097)
    )
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=response)
    context.__aexit__ = AsyncMock(return_value=False)
    provider.mass.http_session.post.return_value = context


@pytest.mark.parametrize("ack_status", ["queued", "processed", "pending_metadata"])
async def test_upload_requires_matching_durable_acknowledgement(ack_status: str) -> None:
    """Only an acknowledgement of the exact submitted document marks delivery."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    extra = await _envelope(provider)
    body = json.dumps(
        decode_document(extra["dejavu"]), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    key = hashlib.sha256(body.encode()).hexdigest()
    _response(provider, 202, {"submission_id": key, "status": ack_status})

    await provider._upload(extra)

    request = provider.mass.http_session.post.call_args
    assert request.args[0] == "http://localhost:8097/api/v1/dejavu/submissions"
    assert request.kwargs["headers"]["Idempotency-Key"] == key
    assert "X-Application-Key" not in request.kwargs["headers"]
    assert request.kwargs["allow_redirects"] is False
    assert "private-provider" not in request.kwargs["data"]
    assert provider.mass.cache.set.call_args.args[1] == {"accepted": True}


async def test_failed_upload_is_retryable_without_reanalysis() -> None:
    """Network failures are retried but invalid acknowledgements stop uploads."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    extra = await _envelope(provider)
    _response(provider, 503, {})
    await provider._upload(extra)
    status = provider.mass.cache.set.call_args.args[1]
    assert "retry_at" in status
    assert not provider._upload_blocked
    provider._retry_at = 0
    provider._next_request_at = 0
    _response(provider, 202, {"submission_id": "incorrect", "status": "queued"})
    await provider._upload(extra)
    assert provider._upload_blocked


async def test_acknowledged_upload_is_not_sent_again() -> None:
    """The destination-scoped acknowledgement prevents repeated delivery."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    extra = await _envelope(provider)
    provider.mass.cache.get.return_value = {"accepted": True}
    await provider._upload(extra)
    provider.mass.http_session.post.assert_not_called()


async def test_private_analysis_is_not_backfilled_after_consent() -> None:
    """New consent does not authorize sharing old private analysis."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    extra = await _envelope(provider)
    extra["upload_destination"] = None
    await provider._upload(extra)
    provider.mass.http_session.post.assert_not_called()


async def test_destination_change_does_not_redirect_old_uploads() -> None:
    """Consent for one server cannot be transferred to another server."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    extra = await _envelope(provider)
    extra["upload_destination"] = "https://old.example/api/v1/dejavu/submissions"
    await provider._upload(extra)
    provider.mass.http_session.post.assert_not_called()


async def test_remote_plain_http_is_rejected() -> None:
    """Remote upload authentication is never sent over plain HTTP."""
    provider = await _provider(upload=True)
    assert isinstance(provider.config, MagicMock)
    provider.config.get_value.side_effect = lambda key: {
        CONF_UPLOAD: True,
        CONF_URL: "http://remote.example",
        CONF_KEY: "test-key",
    }.get(key)
    with pytest.raises(InvalidDataError, match="HTTPS"):
        await provider.handle_async_init()


@pytest.mark.parametrize("application_key", [None, "test-recovery-application-token"])
async def test_recovery_uploads_current_analysis_without_reanalyzing(
    application_key: str | None,
) -> None:
    """Version gating must not prevent recovery of a previously persisted submission."""
    provider = await _provider(upload=True, application_key=application_key)
    assert isinstance(provider.mass, MagicMock)
    extra = await _envelope(provider)
    header, _ = encode(AudioAnalysisData(extra_data=extra))

    async def rows(_domain: str) -> AsyncGenerator[dict[str, bytes]]:
        yield {"header": header.encode("utf-8")}

    provider.mass.streams.audio_analysis.iter_audio_analysis_rows.side_effect = rows
    provider.mass.streams.audio_analysis.get_audio_analysis_version.return_value = 2
    stream = _stream()
    assert not await provider.start_analysis("current", stream, stream.audio_format)
    body = json.dumps(
        decode_document(extra["dejavu"]), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    key = hashlib.sha256(body.encode()).hexdigest()
    _response(provider, 202, {"submission_id": key, "status": "queued"})

    await provider._recover_uploads()

    provider.mass.http_session.post.assert_called_once()
    assert (
        provider.mass.http_session.post.call_args.kwargs["headers"].get("X-Application-Key")
        == application_key
    )
    assert provider.mass.cache.set.call_args.args[1] == {"accepted": True}
    provider.mass.call_later.assert_not_called()


async def test_rejected_authentication_stops_future_uploads() -> None:
    """An authentication rejection must not cause endless retries."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    extra = await _envelope(provider)
    _response(provider, 401, {})
    await provider._upload(extra)
    await provider._upload(extra)
    assert provider._upload_blocked
    provider.mass.http_session.post.assert_called_once()
    provider.mass.call_later.assert_not_called()


async def test_rate_limit_defers_all_submissions() -> None:
    """A server rate limit delays new submissions as well as the rejected payload."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    extra = await _envelope(provider)
    _response(provider, 429, {})
    response = provider.mass.http_session.post.return_value.__aenter__.return_value
    response.headers = {"Retry-After": "3600"}
    await provider._upload(extra)
    await provider._upload(extra)
    provider.mass.http_session.post.assert_called_once()
    assert provider.mass.cache.set.call_args.kwargs["expiration"] == 3600
    assert provider.mass.call_later.call_args.args[0] > 3500


async def test_upload_can_be_revoked_before_request() -> None:
    """Consent is rechecked after asynchronous cache access before sending."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    assert isinstance(provider.config, MagicMock)
    extra = await _envelope(provider)

    async def revoke(*_args: Any, **_kwargs: Any) -> None:
        assert isinstance(provider.config, MagicMock)
        provider.config.get_value.side_effect = lambda key: {CONF_UPLOAD: False}.get(key)

    provider.mass.cache.get.side_effect = revoke
    await provider._upload(extra)
    provider.mass.http_session.post.assert_not_called()


async def test_unload_cancels_recovery_and_releases_buffers() -> None:
    """Provider unload must terminate background recovery and buffered sessions."""
    provider = await _provider(upload=True)
    await _analyze(provider)
    provider._recovery_task = asyncio.create_task(asyncio.sleep(60))
    provider._recovery_timer = asyncio.get_running_loop().call_later(60, lambda: None)
    await provider.unload()
    assert provider._recovery_task.cancelled()
    assert provider._recovery_timer.cancelled()
    assert not provider._data


@pytest.mark.parametrize("application_key", [None, "", "test-fingerprint-application-token"])
async def test_real_http_submission_uses_documented_wire_contract(
    application_key: str | None,
) -> None:
    """A real HTTP request carries only the fingerprint envelope and validates its ack."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    assert isinstance(provider.config, MagicMock)
    received: list[dict[str, Any]] = []

    async def ingest(request: web.Request) -> web.Response:
        raw = await request.read()
        received.append(json.loads(raw))
        canonical = json.dumps(
            decode_document(json.loads(raw)), sort_keys=True, separators=(",", ":")
        )
        key = hashlib.sha256(canonical.encode()).hexdigest()
        assert request.headers["Idempotency-Key"] == key
        assert request.headers["X-API-Key"] == "test-key"
        assert request.headers.get("X-Application-Key") == (application_key or None)
        return web.json_response({"submission_id": key, "status": "queued"}, status=202)

    app = web.Application()

    async def capabilities(_request: web.Request) -> web.Response:
        return web.json_response({"api_version": 1, "fingerprint_encodings": [ENCODING_ID]})

    app.router.add_get("/api/v1/dejavu/capabilities", capabilities)
    app.router.add_post("/api/v1/dejavu/submissions", ingest)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        host, port = runner.addresses[0]
        values = {
            CONF_UPLOAD: True,
            CONF_URL: f"http://{host}:{port}",
            CONF_KEY: "test-key",
            CONF_APPLICATION_KEY: application_key,
        }
        provider.config.get_value.side_effect = lambda key: values.get(key)
        extra = await _envelope(provider)
        async with ClientSession() as session:
            provider.mass.http_session = session
            await provider._upload(extra)
        assert len(received) == 1
        assert set(received[0]) == {
            "format_id",
            "sample_rate",
            "offset_unit",
            "windows",
            "fingerprints",
            "metadata",
            "fingerprints_encoding",
        }
        assert "private-provider" not in json.dumps(received)
        assert not application_key or application_key not in json.dumps(received)
        assert provider.mass.cache.set.call_args.args[1] == {"accepted": True}
    finally:
        await runner.cleanup()


async def test_application_override_is_optional_and_secure() -> None:
    """The additive token field does not change the existing persistent API key."""
    provider = await _provider()
    entries = {entry.key: entry for entry in await provider.get_config_entries()}
    assert entries[CONF_APPLICATION_KEY].type == ConfigEntryType.SECURE_STRING
    assert entries[CONF_APPLICATION_KEY].required is False
    assert entries[CONF_APPLICATION_KEY].default_value is None
    assert entries[CONF_KEY].type == ConfigEntryType.SECURE_STRING


@pytest.mark.parametrize("override", [None, "specific-server-token"])
async def test_central_application_key_and_override(
    monkeypatch: pytest.MonkeyPatch, override: str | None
) -> None:
    """A configured override takes precedence over the shared app-variable resolver."""
    calls: list[str] = []

    def resolve(name: str) -> str:
        calls.append(name)
        return "central-application-token"

    monkeypatch.setattr("music_assistant.providers.dejavu_analysis.app_var", resolve)
    provider = await _provider(upload=True, application_key=override)
    assert isinstance(provider.mass, MagicMock)
    extra = await _envelope(provider)
    body = json.dumps(
        decode_document(extra["dejavu"]), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    identifier = hashlib.sha256(body.encode()).hexdigest()
    _response(provider, 202, {"submission_id": identifier, "status": "queued"})
    await provider._upload(extra)
    request = provider.mass.http_session.post.call_args
    assert request.kwargs["headers"]["X-Application-Key"] == (
        override or "central-application-token"
    )
    assert calls == ([] if override else ["dejavu_application_key"] * 2)
    assert "central-application-token" not in json.dumps(extra)
    assert "specific-server-token" not in json.dumps(extra)
    provider.mass.http_session.post.assert_called_once()


async def test_application_key_uses_existing_environment_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared resolver reads the new local application variable without a bundled key."""
    monkeypatch.setenv("MASS_APP_VAR_DEJAVU_APPLICATION_KEY", "local-application-token")
    monkeypatch.setattr("music_assistant.providers.dejavu_analysis.app_var", app_var)
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    _response(provider, 202, {"submission_id": "test-submission", "status": "queued"})
    await provider._post_document(
        "http://localhost:8097/api/v1/dejavu/submissions", "{}", "test-submission"
    )
    assert provider.mass.http_session.post.call_args.kwargs["headers"]["X-Application-Key"] == (
        "local-application-token"
    )


async def test_sampled_analysis_is_eligible_for_full_reanalysis() -> None:
    """The version bump revisits old sampled analyses without deleting them first."""
    provider = await _provider()
    assert isinstance(provider.mass, MagicMock)
    provider.mass.streams.audio_analysis.get_audio_analysis_version.return_value = 1
    stream = _stream()
    assert await provider.start_analysis("full-upgrade", stream, stream.audio_format)
    assert provider.analysis_version == 2
    provider.mass.streams.audio_analysis.set_audio_analysis.assert_not_awaited()
    await provider.cancel("full-upgrade")


async def test_legacy_pending_upload_is_still_recoverable() -> None:
    """Old persisted sampled documents retain their delivery and consent contract."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    extra = await _envelope(provider)
    extra["dejavu"]["format_id"] = "ma-dejavu-landmarks-v1"
    body = json.dumps(
        decode_document(extra["dejavu"]), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    identifier = hashlib.sha256(body.encode()).hexdigest()
    _response(provider, 202, {"submission_id": identifier, "status": "queued"})
    await provider._upload(extra)
    provider.mass.http_session.post.assert_called_once()
    assert provider.mass.cache.set.call_args.args[1] == {"accepted": True}


async def test_compact_capability_rejection_stops_uploads() -> None:
    """Old receivers fail explicitly before a compact POST is sent."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    context = provider.mass.http_session.get.return_value
    raw = json.dumps({"api_version": 1, "fingerprint_encodings": ["json"]}).encode()
    context.__aenter__.return_value.content.readexactly.side_effect = asyncio.IncompleteReadError(
        raw, 4097
    )
    await provider._upload(await _envelope(provider))
    assert provider._upload_blocked
    provider.mass.http_session.post.assert_not_called()


async def test_capability_and_recovered_uploads_share_request_pacing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The initial capability GET and consecutive recovery POSTs each respect the rate limit."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    extra = await _envelope(provider)
    clock = 10.0
    requests: list[float] = []

    async def sleep(delay: float) -> None:
        nonlocal clock
        clock += delay

    monkeypatch.setattr("music_assistant.providers.dejavu_analysis.time.monotonic", lambda: clock)
    monkeypatch.setattr("music_assistant.providers.dejavu_analysis.asyncio.sleep", sleep)
    get_context = provider.mass.http_session.get.return_value

    def get(*_args: Any, **_kwargs: Any) -> MagicMock:
        requests.append(clock)
        assert isinstance(get_context, MagicMock)
        return get_context

    def post(*_args: Any, **_kwargs: Any) -> MagicMock:
        requests.append(clock)
        assert isinstance(provider.mass, MagicMock)
        context = provider.mass.http_session.post.return_value
        assert isinstance(context, MagicMock)
        return context

    provider.mass.http_session.get.side_effect = get
    provider.mass.http_session.post.side_effect = post
    for title in ("first", "second", "third"):
        extra["dejavu"]["metadata"]["title"] = title
        body = json.dumps(decode_document(extra["dejavu"]), sort_keys=True, separators=(",", ":"))
        identifier = hashlib.sha256(body.encode()).hexdigest()
        _response(provider, 202, {"submission_id": identifier, "status": "queued"})
        await provider._upload(extra)
    assert requests == [10.0, 11.0, 12.0, 13.0]
    provider.mass.http_session.get.assert_called_once()


async def test_compact_upload_retains_legacy_identity() -> None:
    """Recovered legacy results retain their ID while using compact delivery."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    extra = await _envelope(provider)
    extra["dejavu"] = decode_document(extra["dejavu"])
    canonical = json.dumps(extra["dejavu"], sort_keys=True, separators=(",", ":"))
    identifier = hashlib.sha256(canonical.encode()).hexdigest()
    _response(provider, 202, {"submission_id": identifier, "status": "queued"})
    await provider._upload(extra)
    wire = json.loads(provider.mass.http_session.post.call_args.kwargs["data"])
    assert wire["fingerprints_encoding"] == ENCODING_ID
    assert decode_document(wire) == extra["dejavu"]
    assert (
        provider.mass.http_session.post.call_args.kwargs["headers"]["Idempotency-Key"] == identifier
    )


@pytest.mark.parametrize(
    "raw",
    [
        b"not JSON",
        b" " * 4097,
        b'{"api_version":true,"fingerprint_encodings":[]}',
    ],
)
async def test_invalid_capabilities_stop_uploads(raw: bytes) -> None:
    """Malformed and oversized capability responses cannot acknowledge support."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    response = provider.mass.http_session.get.return_value.__aenter__.return_value
    response.content.readexactly.side_effect = asyncio.IncompleteReadError(raw, 4097)
    await provider._upload(await _envelope(provider))
    assert provider._upload_blocked
    provider.mass.http_session.post.assert_not_called()


async def test_consent_revoked_during_capability_check_is_respected() -> None:
    """A support response cannot override consent changed while awaiting the network."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    assert isinstance(provider.config, MagicMock)
    response = provider.mass.http_session.get.return_value.__aenter__.return_value

    async def revoke(_size: int) -> bytes:
        assert isinstance(provider.config, MagicMock)
        provider.config.get_value.side_effect = lambda _key: None
        raw = json.dumps({"api_version": 1, "fingerprint_encodings": [ENCODING_ID]}).encode()
        raise asyncio.IncompleteReadError(raw, 4097)

    response.content.readexactly.side_effect = revoke
    await provider._upload(await _envelope(provider))
    provider.mass.http_session.post.assert_not_called()
    provider.mass.cache.set.assert_not_awaited()


async def test_existing_analysis_migration_preserves_header_payload_and_consent(
    tmp_path: Path,
) -> None:
    """Existing headers compact once without rewriting payloads or granting upload consent."""
    provider = await _provider()
    assert isinstance(provider.mass, MagicMock)
    provider.mass.storage_path = str(tmp_path)
    database = DatabaseConnection(str(tmp_path / AA_DB_FILENAME))
    await database.setup()
    try:
        await create_analysis_tables(database)
        extra = await _envelope(provider)
        extra["dejavu"] = decode_document(extra["dejavu"])
        header, payload = encode(AudioAnalysisData(extra_data=extra, beats=[1.0, 2.0]))
        values = {
            "media_type": "track",
            "item_id": "track",
            "provider": "test",
            "aa_provider_domain": "dejavu_analysis",
            "header": header,
            "payload": payload,
            "analysis_version": 2,
            "timestamp_created": 123,
        }
        await database.insert(AA_TABLE_ANALYSIS, values)
        await database.insert(
            AA_TABLE_ANALYSIS, {**values, "item_id": "broken", "header": "not JSON"}
        )
        controller = AudioAnalysisDatabaseMixin()
        controller._database = database
        controller._database_ready = True
        controller.logger = provider.logger
        provider.mass.streams.audio_analysis = controller
        await provider.loaded_in_mass()
        row = await database.get_row(AA_TABLE_ANALYSIS, {"item_id": "track"})
        assert row is not None
        assert row["payload"] == payload
        assert row["timestamp_created"] == 123
        assert row["analysis_version"] == 2
        converted = json.loads(row["header"])
        assert converted["arrays"] == json.loads(header)["arrays"]
        assert converted["extra_data"]["upload_destination"] is None
        assert decode_document(converted["extra_data"]["dejavu"]) == extra["dejavu"]
        broken = await database.get_row(AA_TABLE_ANALYSIS, {"item_id": "broken"})
        assert broken is not None
        assert broken["header"] == "not JSON"
        marker = await database.get_row(
            AA_TABLE_SETTINGS, {"key": "dejavu_analysis_header_version"}
        )
        assert marker is not None
        assert marker["value"] == "1"
        assert len(list(tmp_path.glob("*.bak"))) == 1
        await provider.loaded_in_mass()
        assert len(list(tmp_path.glob("*.bak"))) == 1
    finally:
        await database.close()
