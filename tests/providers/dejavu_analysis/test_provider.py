"""Fingerprint analysis lifecycle, consent, and upload recovery tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncGenerator
from email.utils import formatdate
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from aiohttp import ClientSession, web
from audio_fingerprint.codec import ENCODING_ID, decode_document
from music_assistant_models.enums import ConfigEntryType, ContentType, ExternalID, MediaType
from music_assistant_models.errors import InvalidDataError
from music_assistant_models.media_items import Album, Artist, AudioFormat, Track
from music_assistant_models.queue_item import QueueItem
from music_assistant_models.streamdetails import StreamDetails
from music_assistant_models.unique_list import UniqueList

from music_assistant.controllers.streams.audio_analysis import AudioAnalysisController
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
    controller = mass.streams.audio_analysis
    controller.mass = mass

    async def resolve(stream: StreamDetails) -> Track | None:
        return await AudioAnalysisController.get_analysis_track(controller, stream)

    controller.get_analysis_track = AsyncMock(side_effect=resolve)
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
    assert track.album is not None
    track.album.add_external_id(ExternalID.MB_ALBUM, "f2148a15-2589-4aa3-8db5-528950f44d20")
    track.album.add_external_id(ExternalID.MB_RELEASEGROUP, "e80e2732-0343-42af-92c7-da4bdf9c30d6")
    item = QueueItem.from_media_item("queue", track)
    item.streamdetails = stream
    provider.mass.player_queues.items.side_effect = lambda *_args: [item]
    assert await provider.start_analysis("session", stream, stream.audio_format)
    metadata = provider._data["session"].metadata
    assert metadata["title"] == "Queued track"
    assert metadata["artists"] == ["Queued artist"]
    assert metadata["album"] == "Queued album"
    assert metadata["isrc"] == "USAAA2600001"
    assert provider._data["session"].release_hints == {
        "release_mbid": "f2148a15-2589-4aa3-8db5-528950f44d20",
        "release_group_mbid": "e80e2732-0343-42af-92c7-da4bdf9c30d6",
    }
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


def _repair_track() -> Track:
    return Track(
        item_id="track",
        provider="library",
        name="Recovered title",
        provider_mappings=set(),
        artists=UniqueList(
            [Artist(item_id="artist", provider="library", name="Artist", provider_mappings=set())]
        ),
        album=Album(item_id="album", provider="library", name="Album", provider_mappings=set()),
        external_ids={(ExternalID.ISRC, "USAAA2600001")},
    )


@pytest.mark.parametrize("duration", [None, True, 0, -1, float("nan"), float("inf")])
def test_metadata_completeness_requires_valid_actual_duration(duration: Any) -> None:
    """Recording identifiers and descriptions cannot substitute for a valid analyzed duration."""
    metadata = DejavuAnalysisProvider._track_metadata(_repair_track())
    metadata["duration"] = duration
    assert not DejavuAnalysisProvider._metadata_complete(metadata)


async def test_recovery_repairs_before_cached_acceptance_and_preserves_acoustics() -> None:
    """An accepted duration-only result gains library metadata without reanalysis or new consent."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    extra = await _envelope(provider)
    extra["dejavu"]["metadata"] = {"duration": 5}
    header, _ = encode(AudioAnalysisData(extra_data=extra))
    old_doc = decode_document(extra["dejavu"])
    old_id = hashlib.sha256(
        json.dumps(old_doc, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    track = _repair_track()
    repaired = provider._repair_header(header.encode(), track)
    assert repaired is not None
    repaired_extra = json.loads(repaired)["extra_data"]
    decoded = decode_document(repaired_extra["dejavu"])
    new_id = hashlib.sha256(
        json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert new_id != old_id
    assert decoded["fingerprints"] == old_doc["fingerprints"]
    assert decoded["windows"] == old_doc["windows"]
    assert decoded["metadata"]["duration"] == 5
    assert repaired_extra["upload_destination"] == extra["upload_destination"]

    async def rows(_domain: str) -> AsyncGenerator[tuple[dict[str, Any], Track]]:
        yield {"id": 1, "header": header.encode()}, track

    provider.mass.streams.audio_analysis.iter_analysis_rows_with_library_tracks.side_effect = rows
    provider.mass.streams.audio_analysis.replace_analysis_header = AsyncMock(return_value=True)
    old_cache_key = hashlib.sha256(f"{extra['upload_destination']}|{old_id}".encode()).hexdigest()

    async def cached(key: str, **_kwargs: Any) -> dict[str, bool] | None:
        return {"accepted": True} if key == old_cache_key else None

    provider.mass.cache.get.side_effect = cached
    _response(provider, 202, {"submission_id": new_id, "status": "queued"})
    await provider._recover_uploads()
    provider.mass.streams.audio_analysis.replace_analysis_header.assert_awaited_once()
    provider.mass.http_session.post.assert_called_once()
    assert provider._repair_header(repaired.encode(), track) is None


@pytest.mark.parametrize(("private", "conflict"), [(True, False), (False, True)])
async def test_metadata_repair_never_uploads_private_or_concurrently_changed_results(
    private: bool, conflict: bool
) -> None:
    """Repair cannot grant consent or overwrite a concurrently replaced analysis."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    extra = await _envelope(provider)
    extra["dejavu"]["metadata"] = {"duration": 5}
    if private:
        extra["upload_destination"] = None
    header, _ = encode(AudioAnalysisData(extra_data=extra))

    async def rows(_domain: str) -> AsyncGenerator[tuple[dict[str, Any], Track]]:
        yield {"id": 1, "header": header.encode()}, _repair_track()

    controller = provider.mass.streams.audio_analysis
    controller.iter_analysis_rows_with_library_tracks.side_effect = rows
    controller.replace_analysis_header = AsyncMock(return_value=not conflict)
    await provider._recover_uploads()
    provider.mass.http_session.post.assert_not_called()
    if private:
        controller.replace_analysis_header.assert_not_awaited()


async def test_repair_preserves_nonempty_metadata_and_unmatched_results() -> None:
    """Existing descriptions are not replaced and unmatched rows are not guessed."""
    provider = await _provider(upload=True)
    extra = await _envelope(provider)
    extra["dejavu"]["metadata"] = {"duration": 5, "title": "Original title"}
    header, _ = encode(AudioAnalysisData(extra_data=extra))
    assert provider._repair_header(header.encode(), None) is None
    repaired = provider._repair_header(header.encode(), _repair_track())
    assert repaired is not None
    assert json.loads(repaired)["extra_data"]["dejavu"]["metadata"]["title"] == "Original title"


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


@pytest.mark.parametrize("limit", [4096, 32768])
@pytest.mark.parametrize("size_delta", [-1, 0, 1])
async def test_response_body_limit(limit: int, size_delta: int) -> None:
    """Accept bodies at the limit and reject even one additional byte."""
    response = MagicMock()
    body = b"x" * (limit + size_delta)
    if size_delta <= 0:
        response.content.readexactly = AsyncMock(
            side_effect=asyncio.IncompleteReadError(body, limit + 1)
        )
        assert await DejavuAnalysisProvider._read_response(response, limit, "too large") == body
    else:
        response.content.readexactly = AsyncMock(return_value=body)
        with pytest.raises(InvalidDataError, match="too large"):
            await DejavuAnalysisProvider._read_response(response, limit, "too large")
    response.content.readexactly.assert_awaited_once_with(limit + 1)


@pytest.mark.parametrize("status", ["full", "partial", "unknown"])
async def test_availability_before_sampler(
    status: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Only verified full coverage skips the sampler."""
    caplog.set_level(logging.DEBUG)
    provider = await _provider(upload=True, application_key="application")
    assert isinstance(provider.mass, MagicMock)
    destination = provider._configured_destination()
    provider._capability_destination = destination
    provider._availability_supported = True
    track = _repair_track()
    provider.mass.streams.audio_analysis.get_analysis_track = AsyncMock(return_value=track)
    _response(
        provider,
        200,
        {
            "version": 1,
            "items": [
                {
                    "index": 0,
                    "status": status,
                    "recording_mbid": (
                        None if status == "unknown" else "e9c2a2a3-4f4b-4b6a-9c31-3a8b2f3c7f10"
                    ),
                }
            ],
        },
    )
    sampler = MagicMock()
    monkeypatch.setattr("music_assistant.providers.dejavu_analysis.WindowSampler", sampler)
    stream = _stream()
    assert await provider.start_analysis("session", stream, stream.audio_format) is (
        status != "full"
    )
    assert sampler.call_count == (0 if status == "full" else 1)
    message = "Skipping fingerprint analysis for Recovered title: already available on server"
    assert (message in caplog.text) is (status == "full")
    assert "private-provider.example" not in caplog.text
    request = provider.mass.http_session.post.call_args
    assert request.kwargs["headers"]["X-API-Key"] == "test-key"
    assert request.kwargs["headers"]["X-Application-Key"] == "application"
    assert provider.analysis_version == 2


@pytest.mark.parametrize(
    ("http_status", "body"),
    [
        (401, {}),
        (429, {}),
        (200, {"version": 1, "items": []}),
        (200, {"version": 2, "items": []}),
        (200, {"version": 1, "items": [{"index": 1, "status": "full", "recording_mbid": None}]}),
        (200, {"version": 1, "items": [{"index": 0, "status": "full", "recording_mbid": "bad"}]}),
    ],
)
async def test_availability_failure_continues_analysis(
    http_status: int, body: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """Failures are logged and do not prevent analysis."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    provider._capability_destination = provider._configured_destination()
    provider._availability_supported = True
    provider.mass.streams.audio_analysis.get_analysis_track = AsyncMock(
        return_value=_repair_track()
    )
    _response(provider, http_status, body)
    stream = _stream()
    assert await provider.start_analysis("session", stream, stream.audio_format)
    assert "availability check failed" in caplog.text


@pytest.mark.parametrize(("http_status", "date_header"), [(429, False), (503, True)])
async def test_availability_pacing_and_cooldown_allow_distinct_tracks(
    http_status: int,
    date_header: bool,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rate limits pause preflights, not PCM analysis or normal uploads."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    provider._capability_destination = provider._configured_destination()
    provider._availability_supported = True
    tracks = [_repair_track(), _repair_track(), _repair_track()]
    tracks[1].name = "Another recording"
    tracks[2].name = "Later recording"
    provider.mass.streams.audio_analysis.get_analysis_track = AsyncMock(side_effect=tracks)
    _response(provider, http_status, {})
    response = provider.mass.http_session.post.return_value.__aenter__.return_value
    response.headers = {
        "Retry-After": formatdate(time.time() + 3600, usegmt=True) if date_header else "3600"
    }
    pace = AsyncMock()
    monkeypatch.setattr(provider, "_pace_request", pace)
    stream = _stream()
    assert await provider.start_analysis("first", stream, stream.audio_format)
    assert 3598 <= provider._availability_retry_at - time.monotonic() <= 3600
    assert provider._retry_at == 0
    assert await provider.start_analysis("second", stream, stream.audio_format)
    assert provider.mass.http_session.post.call_count == 1
    pace.assert_awaited_once()
    assert "availability check failed" in caplog.text
    provider._availability_retry_at = 0
    assert await provider.start_analysis("third", stream, stream.audio_format)
    assert provider.mass.http_session.post.call_count == 2
    assert pace.await_count == 2


async def test_retry_after_minimum_invalid_and_ceiling(caplog: pytest.LogCaptureFixture) -> None:
    """Retry headers retain the existing safe limits and invalid-value logging."""
    provider = await _provider(upload=True)
    assert provider._retry_delay({"Retry-After": "1"}) == 300
    assert provider._retry_delay({"Retry-After": "999999"}) == 86400
    assert provider._retry_delay({"Retry-After": "invalid"}) == 300
    assert "invalid Retry-After" in caplog.text


async def test_recovery_checks_batch_without_accepting_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovered full references are skipped without marking them accepted."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    destination = provider._configured_destination()
    assert destination
    provider._capability_destination = destination
    provider._availability_supported = True
    metadata = {**provider._track_metadata(_repair_track()), "duration": 5}
    extras = [
        {"upload_destination": destination, "dejavu": {"metadata": metadata}} for _ in range(3)
    ]
    _response(
        provider,
        200,
        {
            "version": 1,
            "items": [
                {
                    "index": i,
                    "status": status,
                    "recording_mbid": (
                        None if status == "unknown" else "e9c2a2a3-4f4b-4b6a-9c31-3a8b2f3c7f10"
                    ),
                }
                for i, status in enumerate(("full", "partial", "unknown"))
            ],
        },
    )
    upload = AsyncMock()
    monkeypatch.setattr(provider, "_upload", upload)
    await provider._upload_recovered(extras, destination)
    assert provider.mass.http_session.post.call_count == 1
    assert upload.await_count == 2
    assert len(provider.mass.http_session.post.call_args.kwargs["json"]["items"]) == 3
    assert all(
        call.args[1] != {"accepted": True} for call in provider.mass.cache.set.await_args_list
    )


async def test_availability_cache_is_scoped_to_consent_destination_and_credentials() -> None:
    """Cached availability never crosses a destination or credential boundary."""
    provider = await _provider(upload=True, application_key="application")
    assert isinstance(provider.mass, MagicMock)
    assert isinstance(provider.config, MagicMock)
    values = {
        CONF_UPLOAD: True,
        CONF_URL: "http://localhost:8097",
        CONF_KEY: "first",
        CONF_APPLICATION_KEY: "application",
    }
    provider.config.get_value.side_effect = lambda key, *_args: values.get(key)
    cache: dict[str, Any] = {}
    provider.mass.cache.get.side_effect = lambda key, **_kwargs: cache.get(key)
    provider.mass.cache.set.side_effect = lambda key, value, **_kwargs: cache.update({key: value})
    destination = provider._configured_destination()
    assert destination
    provider._capability_destination = destination
    provider._availability_supported = True
    _response(
        provider,
        200,
        {
            "version": 1,
            "items": [
                {
                    "index": 0,
                    "status": "full",
                    "recording_mbid": "e9c2a2a3-4f4b-4b6a-9c31-3a8b2f3c7f10",
                }
            ],
        },
    )
    metadata = {**provider._track_metadata(_repair_track()), "duration": 5}
    assert await provider._availability(destination, [metadata]) == ["full"]
    assert await provider._availability(destination, [metadata]) == ["full"]
    assert provider.mass.http_session.post.call_count == 1
    values[CONF_KEY] = "second"
    assert await provider._availability(destination, [metadata]) == ["full"]
    assert provider.mass.http_session.post.call_count == 2
    values[CONF_URL] = "http://localhost:8098"
    new_destination = provider._configured_destination()
    assert new_destination
    provider._capability_destination = new_destination
    assert await provider._availability(new_destination, [metadata]) == ["full"]
    assert provider.mass.http_session.post.call_count == 3
    values[CONF_UPLOAD] = False
    assert await provider._availability(new_destination, [metadata]) == ["unknown"]
    assert provider.mass.http_session.post.call_count == 3


async def test_old_server_without_availability_is_optional() -> None:
    """An older compatible receiver does not cause a failed preflight."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    destination = provider._configured_destination()
    assert destination
    metadata = {**provider._track_metadata(_repair_track()), "duration": 5}
    assert await provider._availability(destination, [metadata]) == ["unknown"]
    provider.mass.http_session.post.assert_not_called()
    assert not provider._upload_blocked


async def test_recovery_pass_batches_and_retains_full_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recovery pass checks eligible rows together without replacing complete headers."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    extra = await _envelope(provider)
    extra["dejavu"]["metadata"] = {**provider._track_metadata(_repair_track()), "duration": 5}
    header, _ = encode(AudioAnalysisData(extra_data=extra))

    async def rows(_domain: str) -> AsyncGenerator[tuple[dict[str, Any], Track]]:
        for index in range(3):
            yield {"id": index, "header": header.encode()}, _repair_track()

    controller = provider.mass.streams.audio_analysis
    controller.iter_analysis_rows_with_library_tracks.side_effect = rows
    controller.replace_analysis_header = AsyncMock()
    availability = AsyncMock(return_value=["full", "partial", "unknown"])
    upload = AsyncMock()
    monkeypatch.setattr(provider, "_availability", availability)
    monkeypatch.setattr(provider, "_upload", upload)
    await provider._recover_uploads()
    assert availability.await_args is not None
    assert len(availability.await_args.args[1]) == 3
    assert upload.await_count == 2
    controller.replace_analysis_header.assert_not_awaited()


async def test_availability_propagates_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancelled preflight does not silently continue analysis."""
    provider = await _provider(upload=True)
    destination = provider._configured_destination()
    assert destination
    monkeypatch.setattr(
        provider, "_check_capabilities", AsyncMock(side_effect=asyncio.CancelledError)
    )
    metadata = {**provider._track_metadata(_repair_track()), "duration": 5}
    with pytest.raises(asyncio.CancelledError):
        await provider._availability(destination, [metadata])


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

    async def rows(_domain: str) -> AsyncGenerator[tuple[dict[str, bytes], None]]:
        yield {"header": header.encode("utf-8")}, None

    provider.mass.streams.audio_analysis.iter_analysis_rows_with_library_tracks.side_effect = rows
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


@pytest.mark.parametrize("version", [None, True, 1, 2])
async def test_release_hints_require_explicit_receiver_support(version: Any) -> None:
    """Only receivers advertising version 1 see hints in the canonical upload."""
    provider = await _provider(upload=True)
    assert isinstance(provider.mass, MagicMock)
    extra = await _envelope(provider)
    hints = {"release_mbid": "f2148a15-2589-4aa3-8db5-528950f44d20"}
    extra["release_hints"] = hints
    capabilities: dict[str, Any] = {"api_version": 1, "fingerprint_encodings": [ENCODING_ID]}
    if version is not None:
        capabilities["release_hints_version"] = version
    response = provider.mass.http_session.get.return_value.__aenter__.return_value
    response.content.readexactly.side_effect = asyncio.IncompleteReadError(
        json.dumps(capabilities).encode(), 4097
    )
    expected = decode_document(extra["dejavu"])
    if type(version) is int and version == 1:
        expected["metadata"] = {**expected["metadata"], **hints}
    canonical = json.dumps(expected, sort_keys=True, separators=(",", ":"))
    identifier = hashlib.sha256(canonical.encode()).hexdigest()
    _response(provider, 202, {"submission_id": identifier, "status": "queued"})
    await provider._upload(extra)
    request = provider.mass.http_session.post.call_args
    assert decode_document(json.loads(request.kwargs["data"])) == expected
    assert request.kwargs["headers"]["Idempotency-Key"] == identifier
    assert "release_mbid" not in extra["dejavu"]["metadata"]
    provider.mass.http_session.get.assert_called_once()


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
