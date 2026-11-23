"""Landmark audio analysis with consent-scoped, recoverable fingerprint delivery."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

import aiohttp
from audio_fingerprint import (
    FORMAT_ID,
    LEGACY_FORMAT_ID,
    MAX_DOCUMENT_BYTES,
    MAX_DURATION,
    MAX_TRACK_HASHES,
    SAMPLE_RATE,
    WindowSampler,
)
from audio_fingerprint.codec import ENCODING_ID, decode_document, encode_document
from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import ConfigEntryType, ExternalID, MediaType
from music_assistant_models.errors import InvalidDataError
from yarl import URL

from music_assistant.controllers.streams.audio_analysis_codec import decode_extra_data
from music_assistant.helpers.app_vars import app_var
from music_assistant.models.audio_analysis import AudioAnalysisData, AudioAnalysisError
from music_assistant.models.audio_analysis_provider import AudioAnalysisProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.media_items import AudioFormat, Track
    from music_assistant_models.provider import ProviderManifest
    from music_assistant_models.streamdetails import StreamDetails

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

CONF_UPLOAD = "upload_enabled"
CONF_URL = "upload_url"
CONF_KEY = "api_key"
CONF_APPLICATION_KEY = "application_key"
MAX_RESULT_BYTES = MAX_DOCUMENT_BYTES
MAX_RESULT_HASHES = MAX_TRACK_HASHES
RETRY_SECONDS = 300


@dataclass
class UploadDeferred(Exception):
    """A transient server response with its minimum retry delay."""

    status: int
    retry_seconds: int


@dataclass
class FingerprintSession:
    """Provider-local state for one bounded analysis."""

    sampler: WindowSampler
    destination: str | None
    metadata: dict[str, Any]
    hashes: list[tuple[str, int]] = field(default_factory=list)
    release_hints: dict[str, str] = field(default_factory=dict)


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Create the fingerprint analysis provider."""
    return DejavuAnalysisProvider(mass, manifest, config)


class DejavuAnalysisProvider(AudioAnalysisProvider):
    """Produce local fingerprints and optionally deliver them to a recognition server."""

    max_analysis_duration = MAX_DURATION
    analysis_version = 2

    async def handle_async_init(self) -> None:
        """Prepare bounded processing and validate the configured upload destination."""
        self._data: dict[str, FingerprintSession] = {}
        self._upload_lock = asyncio.Lock()
        self._recovery_task: asyncio.Task[None] | None = None
        self._recovery_timer: asyncio.TimerHandle | None = None
        self._upload_blocked = False
        self._retry_at = 0.0
        self._availability_retry_at = 0.0
        self._next_request_at = 0.0
        self._capability_destination: str | None = None
        self._availability_supported = False
        self._release_hints_supported = False
        self._destination = self._configured_destination()
        if self._destination and not self.config.get_value(CONF_KEY):
            raise InvalidDataError("An API key is required when sharing fingerprints")

    async def loaded_in_mass(self) -> None:
        """Recover eligible uploads after the analysis database is available."""
        if not await self.mass.streams.audio_analysis.migrate_provider_headers(
            self.domain, 1, self._compact_header
        ):
            self._upload_blocked = True
            return
        if self._destination:
            self._start_recovery()

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return explicit sharing consent and server configuration."""
        return (
            ConfigEntry(key=CONF_UPLOAD, type=ConfigEntryType.BOOLEAN, default_value=False),
            ConfigEntry(key=CONF_URL, type=ConfigEntryType.STRING, required=False),
            ConfigEntry(key=CONF_KEY, type=ConfigEntryType.SECURE_STRING, required=False),
            ConfigEntry(
                key=CONF_APPLICATION_KEY, type=ConfigEntryType.SECURE_STRING, required=False
            ),
        )

    async def process_pcm_chunk(self, session_id: str, pcm_chunk: bytes) -> None:
        """Extract fingerprints from the server's already-decoded playback PCM."""
        if (data := self._data.get(session_id)) is None:
            return
        try:
            hashes = await self._run_offloaded(data.sampler.feed, pcm_chunk)
        except ValueError as err:
            raise AudioAnalysisError(str(err)) from err
        data.hashes.extend(hashes)
        if len(data.hashes) > MAX_RESULT_HASHES:
            raise AudioAnalysisError("Fingerprint count exceeds the analysis limit")

    async def cancel(self, session_id: str) -> None:
        """Discard interrupted analysis without uploading partial windows."""
        self._data.pop(session_id, None)
        await super().cancel(session_id)

    async def unload(self, is_removed: bool = False) -> None:
        """Stop upload recovery and release analysis sessions."""
        if self._recovery_timer is not None:
            self._recovery_timer.cancel()
        if self._recovery_task is not None:
            self._recovery_task.cancel()
            await asyncio.gather(self._recovery_task, return_exceptions=True)
        await super().unload(is_removed)
        self._data.clear()

    async def post_analysis(
        self, streamdetails: StreamDetails, analysis: AudioAnalysisData
    ) -> None:
        """Attempt delivery only after the replayable analysis result has been persisted."""
        if analysis.extra_data is not None:
            await self._upload(analysis.extra_data)

    async def _start_analysis(
        self, session_id: str, streamdetails: StreamDetails, audio_format: AudioFormat
    ) -> bool:
        """Accept finite tracks starting at zero and retain only planned windows."""
        if streamdetails.media_type != MediaType.TRACK or streamdetails.seek_position:
            self.logger.debug("Skipping non-track or seeked fingerprint analysis")
            return False
        if not streamdetails.duration or not math.isfinite(streamdetails.duration):
            self.logger.debug("Skipping fingerprint analysis without a finite duration")
            return False
        metadata: dict[str, Any] = {"duration": streamdetails.duration}
        release_hints: dict[str, str] = {}
        destination = self._configured_destination()
        if destination:
            # Only existing library/queue metadata is consulted; this never fetches provider
            # audio or triggers online enrichment. Stream URLs and provider IDs are not submitted.
            track = await self.mass.streams.audio_analysis.get_analysis_track(streamdetails)
            if track is not None:
                metadata.update(self._track_metadata(track))
                if track.album:
                    if release := track.album.mbid:
                        release_hints["release_mbid"] = release
                    if group := track.album.get_external_id(ExternalID.MB_RELEASEGROUP):
                        release_hints["release_group_mbid"] = group
            elif streamdetails.stream_metadata is not None:
                metadata.update(
                    title=streamdetails.stream_metadata.title,
                    artists=[streamdetails.stream_metadata.artist]
                    if streamdetails.stream_metadata.artist
                    else [],
                    album=streamdetails.stream_metadata.album,
                )
            if not self._metadata_complete(metadata):
                self.logger.warning(
                    "Fingerprint metadata lacks required fields or a recording identifier; the receiver must "
                    "hold this contribution pending metadata"
                )
            if (await self._availability(destination, [metadata]))[0] == "full":
                self.logger.debug(
                    "Skipping fingerprint analysis for %s: already available on server",
                    metadata.get("title") or "unknown track",
                )
                return False
        try:
            sampler = WindowSampler(
                streamdetails.duration,
                audio_format.sample_rate,
                audio_format.channels,
                f"pcm_{audio_format.content_type.value}",
            )
        except ValueError as err:
            raise AudioAnalysisError(str(err)) from err
        self._data[session_id] = FingerprintSession(
            sampler, destination, metadata, release_hints=release_hints
        )
        return True

    async def _finalize(self, session_id: str) -> AudioAnalysisData | None:
        """Return completed sampled coverage in a versioned, replayable envelope."""
        data = self._data.pop(session_id, None)
        if data is None:
            return None
        try:
            data.hashes.extend(await self._run_offloaded(data.sampler.feed, b"", last=True))
        except ValueError as err:
            raise AudioAnalysisError(str(err)) from err
        if data.sampler.completed != list(data.sampler.windows):
            raise AudioAnalysisError("Fingerprint analysis did not reach all planned windows")
        actual_duration = data.sampler.position / SAMPLE_RATE
        if abs(actual_duration - data.metadata["duration"]) > 1 / SAMPLE_RATE:
            self.logger.debug(
                "Fingerprint duration corrected from %.6fs to decoded %.6fs",
                data.metadata["duration"],
                actual_duration,
            )
        data.metadata["duration"] = actual_duration
        if not data.hashes:
            raise AudioAnalysisError("No landmark fingerprints found")
        if len(data.hashes) > MAX_RESULT_HASHES:
            raise AudioAnalysisError("Fingerprint count exceeds the analysis limit")
        document = await self._run_offloaded(self._build_document, data)
        decoded = await self._run_offloaded(decode_document, document)
        body = await self._run_offloaded(
            json.dumps, decoded, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if len(body.encode("utf-8")) > MAX_RESULT_BYTES:
            raise AudioAnalysisError("Fingerprint result exceeds the storage limit")
        extra = {"dejavu": document, "upload_destination": data.destination}
        if data.release_hints:
            # Keep local hints outside the document so older plugins can still replay it.
            extra["release_hints"] = data.release_hints
        return AudioAnalysisData(
            duration=data.metadata["duration"],
            extra_data=extra,
        )

    @staticmethod
    def _build_document(data: FingerprintSession) -> dict[str, Any]:
        """Build the submission document from completed sampled coverage."""
        return encode_document(
            {
                "format_id": FORMAT_ID,
                "sample_rate": SAMPLE_RATE,
                "offset_unit": "fft_hops",
                "windows": [
                    {"start_sample": window.start_sample, "sample_count": window.sample_count}
                    for window in data.sampler.completed
                ],
                "fingerprints": [list(pair) for pair in sorted(set(data.hashes))],
                "metadata": data.metadata,
            }
        )

    def _configured_destination(self) -> str | None:
        """Validate a consent-enabled submission URL without exposing credentials."""
        if not self.config.get_value(CONF_UPLOAD):
            return None
        value = self.config.get_value(CONF_URL)
        if not isinstance(value, str) or not value:
            raise InvalidDataError("A recognition server URL is required")
        url = URL(value)
        local_http = url.scheme == "http" and url.host in ("localhost", "127.0.0.1", "::1")
        if (
            not url.host
            or (url.scheme != "https" and not local_http)
            or url.user is not None
            or url.query_string
            or url.fragment
        ):
            raise InvalidDataError("Use HTTPS, without URL credentials, query or fragment")
        return str(url).rstrip("/") + "/api/v1/dejavu/submissions"

    def _start_recovery(self) -> None:
        """Start one managed recovery pass while the provider remains active."""
        if not self.unloading and not self._upload_blocked:
            self._recovery_task = self.mass.create_task(self._recover_uploads())

    async def _recover_uploads(self) -> None:
        """Retry stored consent-eligible results, without reanalyzing tracks."""
        try:
            pending: list[dict[str, Any]] = []
            async for (
                row,
                track,
            ) in self.mass.streams.audio_analysis.iter_analysis_rows_with_library_tracks(
                self.domain
            ):
                try:
                    extra = await self._run_offloaded(decode_extra_data, row["header"])
                except (ValueError, UnicodeError) as err:
                    self.logger.warning("Cannot decode stored fingerprints: %s", err)
                    continue
                if extra is not None:
                    destination = self._configured_destination()
                    if destination is None or extra.get("upload_destination") != destination:
                        continue
                    try:
                        header = await self._run_offloaded(
                            self._repair_header, row["header"], track
                        )
                    except (ValueError, TypeError, UnicodeError) as err:
                        self.logger.warning("Cannot repair stored fingerprint metadata: %s", err)
                        continue
                    if header is not None:
                        if not await self.mass.streams.audio_analysis.replace_analysis_header(
                            row["id"], self.domain, row["header"], header
                        ):
                            self.logger.warning(
                                "Fingerprint analysis changed during metadata repair; deferring upload"
                            )
                            continue
                        extra = await self._run_offloaded(decode_extra_data, header)
                        if extra is None:
                            self.logger.error("Repaired fingerprint header cannot be decoded")
                            continue
                    document = extra.get("dejavu")
                    if isinstance(document, dict) and isinstance(document.get("metadata"), dict):
                        if not self._metadata_complete(document["metadata"]):
                            self.logger.warning(
                                "Stored fingerprint metadata remains incomplete; "
                                "no complete identity-matched library metadata is available"
                            )
                    pending.append(extra)
                    if len(pending) == 100:
                        await self._upload_recovered(pending, destination)
                        pending = []
                if self._upload_blocked or self.unloading or self._retry_at > time.time():
                    return
            if pending and (destination := self._configured_destination()):
                await self._upload_recovered(pending, destination)
        finally:
            if self._retry_at > time.time():
                self._schedule_retry()

    def _schedule_retry(self) -> None:
        """Schedule a recovery pass only while uploads are actually pending."""
        if not self.unloading and not self._upload_blocked:
            self._recovery_timer = self.mass.call_later(
                max(RETRY_SECONDS, self._retry_at - time.time()),
                self._start_recovery,
                task_id=f"{self.instance_id}_fingerprint_recovery",
            )

    async def _upload_recovered(self, extras: list[dict[str, Any]], destination: str) -> None:
        extras = [extra for extra in extras if extra.get("upload_destination") == destination]
        if not extras:
            return
        metadata = [
            extra.get("dejavu", {}).get("metadata", {})
            if isinstance(extra.get("dejavu"), dict)
            else {}
            for extra in extras
        ]
        statuses = await self._availability(destination, metadata)
        for extra, status in zip(extras, statuses, strict=True):
            if status != "full":
                await self._upload(extra, availability_checked=True)
            if self._upload_blocked or self.unloading or self._retry_at > time.time():
                return

    async def _upload(self, extra: dict[str, Any], availability_checked: bool = False) -> None:
        """Deliver one eligible result with idempotency and explicit failure handling."""
        destination = self._configured_destination()
        document = extra.get("dejavu")
        if (
            not availability_checked
            and destination
            and extra.get("upload_destination") == destination
            and isinstance(document, dict)
            and isinstance(document.get("metadata"), dict)
            and (await self._availability(destination, [document["metadata"]]))[0] == "full"
        ):
            return
        async with self._upload_lock:
            destination = self._configured_destination()
            if (
                self.unloading
                or self._upload_blocked
                or self._retry_at > time.time()
                or destination is None
                or extra.get("upload_destination") != destination
            ):
                return
            document = extra.get("dejavu")
            if not isinstance(document, dict) or document.get("format_id") not in (
                FORMAT_ID,
                LEGACY_FORMAT_ID,
            ):
                self.logger.error("Cannot upload an unsupported fingerprint envelope")
                return
            cache_key: str | None = None
            retry_delay = RETRY_SECONDS
            try:
                decoded = await self._upload_document(
                    destination, document, extra.get("release_hints")
                )
                body = await self._run_offloaded(
                    json.dumps, decoded, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                if len(body.encode("utf-8")) > MAX_RESULT_BYTES:
                    self.logger.error("Stored fingerprint upload exceeds the size limit")
                    return
                submission_id = hashlib.sha256(body.encode("utf-8")).hexdigest()
                cache_key = hashlib.sha256(f"{destination}|{submission_id}".encode()).hexdigest()
                status = await self.mass.cache.get(
                    cache_key, provider=self.domain, allow_bypass=False
                )
                if isinstance(status, dict):
                    if status.get("accepted") is True:
                        return
                    if status.get("retry_at", 0) > time.time():
                        self._retry_at = status["retry_at"]
                        self._schedule_retry()
                        return
                if self.unloading or self._configured_destination() != destination:
                    return
                await self._check_capabilities(destination)
                compact = await self._run_offloaded(encode_document, decoded)
                compact_body = await self._run_offloaded(
                    json.dumps, compact, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                if self.unloading or self._configured_destination() != destination:
                    return
                if not await self._post_document(destination, compact_body, submission_id):
                    return
            except UploadDeferred as err:
                retry_delay = err.retry_seconds
                self.logger.warning("Fingerprint upload deferred: HTTP %s", err.status)
            except InvalidDataError as err:
                self._upload_blocked = True
                self.logger.error("Fingerprint uploads stopped: %s", err)
                return
            except (KeyError, ValueError, TypeError, UnicodeError) as err:
                self.logger.error("Cannot encode stored fingerprint upload: %s", err)
                return
            except (TimeoutError, aiohttp.ClientError) as err:
                self.logger.warning("Fingerprint upload failed: %s", type(err).__name__)
            else:
                await self.mass.cache.set(
                    cache_key,
                    {"accepted": True},
                    provider=self.domain,
                    expiration=365 * 86400,
                )
                return
            self._retry_at = time.time() + retry_delay
            if cache_key is not None:
                await self.mass.cache.set(
                    cache_key,
                    {"retry_at": self._retry_at},
                    provider=self.domain,
                    expiration=retry_delay,
                )
            self._schedule_retry()

    async def _upload_document(
        self, destination: str, document: dict[str, Any], hints: object
    ) -> dict[str, Any]:
        """Return the document supported by this receiver for hashing and delivery."""
        decoded = await self._run_offloaded(decode_document, document)
        if not isinstance(hints, dict) or not hints:
            return decoded
        await self._check_capabilities(destination)
        if not self._release_hints_supported:
            return decoded
        metadata = dict(decoded["metadata"])
        for key in ("release_mbid", "release_group_mbid"):
            if isinstance(value := hints.get(key), str):
                try:
                    metadata[key] = str(UUID(value))
                except ValueError:
                    self.logger.warning("Ignoring invalid MusicBrainz upload hint %s", key)
        decoded["metadata"] = metadata
        return decoded

    async def _check_capabilities(self, destination: str) -> None:
        if self._capability_destination == destination:
            return
        headers = await self._upload_headers()
        await self._pace_request()
        if self.unloading or self._configured_destination() != destination:
            return
        async with self.mass.http_session.get(
            destination.rsplit("/", 1)[0] + "/capabilities",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=20),
            allow_redirects=False,
        ) as response:
            if response.status == 429 or response.status >= 500:
                raise UploadDeferred(response.status, self._retry_delay(response.headers))
            if response.status != 200:
                raise InvalidDataError(
                    f"Fingerprint capability check returned HTTP {response.status}; "
                    "update the receiver and gateway before enabling compact uploads"
                )
            raw = await self._read_response(
                response, 4096, "Fingerprint capabilities exceed the size limit"
            )
            try:
                capabilities = json.loads(raw)
            except (ValueError, UnicodeError) as err:
                raise InvalidDataError("Invalid fingerprint capabilities") from err
            if (
                not isinstance(capabilities, dict)
                or type(capabilities.get("api_version")) is not int
                or capabilities.get("api_version") != 1
                or not isinstance(capabilities.get("fingerprint_encodings"), list)
                or ENCODING_ID not in capabilities["fingerprint_encodings"]
            ):
                raise InvalidDataError("Receiver does not support compact fingerprint uploads")
        self._capability_destination = destination
        self._availability_supported = (
            type(capabilities.get("availability_version")) is int
            and capabilities["availability_version"] == 1
        )
        self._release_hints_supported = (
            type(capabilities.get("release_hints_version")) is int
            and capabilities["release_hints_version"] == 1
        )

    async def _availability(self, destination: str, metadata: list[dict[str, Any]]) -> list[str]:
        """Check published coverage, falling back to normal analysis on any remote failure."""
        results = ["unknown"] * len(metadata)
        if self.unloading or self._configured_destination() != destination:
            return results
        pairs = [
            (position, item)
            for position, source in enumerate(metadata)
            if (item := self._availability_identity(source)) is not None
        ]
        positions = [position for position, _ in pairs]
        items = [item for _, item in pairs]
        if not items:
            return results
        try:
            async with self._upload_lock:
                if self._availability_retry_at > time.monotonic():
                    return results
                headers = await self._upload_headers()
                credential = hashlib.sha256(
                    json.dumps(headers, sort_keys=True).encode()
                ).hexdigest()
                keys = [
                    hashlib.sha256(
                        json.dumps([1, destination, credential, item], sort_keys=True).encode()
                    ).hexdigest()
                    for item in items
                ]
                missing: list[int] = []
                cached_results = await asyncio.gather(
                    *(
                        self.mass.cache.get(key, provider=self.domain, allow_bypass=False)
                        for key in keys
                    )
                )
                for index, cached in enumerate(cached_results):
                    if cached in ("full", "partial", "unknown"):
                        results[positions[index]] = cached
                    else:
                        missing.append(index)
                if not missing:
                    if (
                        not self.unloading
                        and self._configured_destination() == destination
                        and await self._upload_headers() == headers
                    ):
                        return results
                    return ["unknown"] * len(metadata)
                await self._check_capabilities(destination)
                if not self._availability_supported:
                    return ["unknown"] * len(metadata)
                if self.unloading or self._configured_destination() != destination:
                    return ["unknown"] * len(metadata)
                statuses = await self._request_availability(
                    destination, headers, [items[index] for index in missing]
                )
                if (
                    self.unloading
                    or self._configured_destination() != destination
                    or await self._upload_headers() != headers
                ):
                    return ["unknown"] * len(metadata)
                for index, status in zip(missing, statuses, strict=True):
                    results[positions[index]] = status
                await asyncio.gather(
                    *(
                        self.mass.cache.set(
                            keys[index],
                            status,
                            provider=self.domain,
                            expiration=300 if status == "full" else 60,
                        )
                        for index, status in zip(missing, statuses, strict=True)
                    )
                )
                if (
                    self.unloading
                    or self._configured_destination() != destination
                    or await self._upload_headers() != headers
                ):
                    return ["unknown"] * len(metadata)
        except UploadDeferred as err:
            self._availability_retry_at = time.monotonic() + err.retry_seconds
            self.logger.warning("Fingerprint availability check failed: HTTP %s", err.status)
            return ["unknown"] * len(metadata)
        except (
            InvalidDataError,
            TimeoutError,
            aiohttp.ClientError,
            ValueError,
            TypeError,
            UnicodeError,
        ) as err:
            self.logger.warning("Fingerprint availability check failed: %s", err)
        return results

    @staticmethod
    def _availability_identity(source: Any) -> dict[str, Any] | None:
        if not isinstance(source, dict):
            return None
        mbid = source.get("recording_mbid")
        isrc = source.get("isrc")
        if mbid and not isinstance(mbid, str):
            return None
        try:
            mbid = str(UUID(mbid)) if isinstance(mbid, str) else None
        except ValueError:
            return None
        if not isinstance(isrc, str) or not re.fullmatch(
            r"[A-Za-z]{2}[A-Za-z0-9]{3}[0-9]{7}", isrc
        ):
            if isrc:
                return None
            isrc = None
        if not mbid and not isrc:
            return None
        title, artists, duration, album = (
            source.get(key) for key in ("title", "artists", "duration", "album")
        )
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(duration)
            or not 0 < duration <= MAX_DURATION
        ):
            duration = None
        if (
            (isinstance(title, str) and len(title) > 500)
            or (isinstance(album, str) and len(album) > 500)
            or (
                isinstance(artists, list)
                and (len(artists) > 30 or any(isinstance(x, str) and len(x) > 500 for x in artists))
            )
        ):
            return None
        item = {
            "format_id": FORMAT_ID,
            "recording_mbid": mbid,
            "isrc": isrc,
            "title": title if isinstance(title, str) else None,
            "album": album if isinstance(album, str) else None,
            "artists": [x for x in artists if isinstance(x, str)]
            if isinstance(artists, list)
            else [],
            "duration": duration,
        }
        return item if len(json.dumps(item).encode()) <= 1000 else None

    async def _request_availability(
        self, destination: str, headers: dict[str, str], items: list[dict[str, Any]]
    ) -> list[str]:
        payload = {
            "version": 1,
            "items": [{**item, "index": number} for number, item in enumerate(items)],
        }
        await self._pace_request()
        if (
            self.unloading
            or self._configured_destination() != destination
            or await self._upload_headers() != headers
        ):
            raise InvalidDataError("Availability credentials or destination changed")
        async with self.mass.http_session.post(
            destination.rsplit("/", 1)[0] + "/availability",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
            allow_redirects=False,
        ) as response:
            if response.status == 429 or response.status >= 500:
                raise UploadDeferred(response.status, self._retry_delay(response.headers))
            if response.status != 200:
                raise InvalidDataError(f"HTTP {response.status}")
            raw = await self._read_response(
                response, 32768, "Availability response exceeds the size limit"
            )
            data = json.loads(raw)
        if (
            not isinstance(data, dict)
            or set(data) != {"version", "items"}
            or type(data["version"]) is not int
            or data["version"] != 1
            or not isinstance(data["items"], list)
            or len(data["items"]) != len(items)
        ):
            raise InvalidDataError("Invalid availability response")
        statuses: list[str] = []
        for number, entry in enumerate(data["items"]):
            if (
                not isinstance(entry, dict)
                or set(entry) != {"index", "status", "recording_mbid"}
                or type(entry["index"]) is not int
                or entry["index"] != number
                or entry["status"] not in ("full", "partial", "unknown")
                or (entry["status"] != "unknown" and not isinstance(entry["recording_mbid"], str))
                or (entry["status"] == "unknown" and entry["recording_mbid"] is not None)
            ):
                raise InvalidDataError("Invalid availability result")
            if entry["recording_mbid"] is not None:
                resolved = str(UUID(entry["recording_mbid"]))
                if items[number]["recording_mbid"] and items[number]["recording_mbid"] != resolved:
                    raise InvalidDataError("Availability recording identifier mismatch")
            statuses.append(entry["status"])
        return statuses

    @staticmethod
    async def _read_response(
        response: aiohttp.ClientResponse, limit: int, oversized_message: str
    ) -> bytes:
        """Read a response body up to its byte limit."""
        try:
            raw = await response.content.readexactly(limit + 1)
        except asyncio.IncompleteReadError as err:
            raw = err.partial
        if len(raw) > limit:
            raise InvalidDataError(oversized_message)
        return raw

    async def _pace_request(self) -> None:
        """Space outbound requests on the serialized upload path at least one second apart."""
        if (delay := self._next_request_at - time.monotonic()) > 0:
            await asyncio.sleep(delay)
        self._next_request_at = time.monotonic() + 1

    def _retry_delay(self, headers: Any) -> int:
        value = headers.get("Retry-After", "")
        try:
            if value.isdigit():
                return max(RETRY_SECONDS, min(int(value), 86400))
            if value:
                seconds = parsedate_to_datetime(value).timestamp() - time.time()
                return max(RETRY_SECONDS, min(math.ceil(seconds), 86400))
        except ValueError, TypeError, OverflowError:
            self.logger.warning("Fingerprint server returned invalid Retry-After")
        return RETRY_SECONDS

    async def _upload_headers(self) -> dict[str, str]:
        application_key = self.config.get_value(CONF_APPLICATION_KEY)
        if not application_key:
            application_key = await self._run_offloaded(app_var, "dejavu_application_key")
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": str(self.config.get_value(CONF_KEY)),
        }
        if application_key:
            if not isinstance(application_key, str):
                raise InvalidDataError("The fingerprint application token must be a string")
            headers["X-Application-Key"] = application_key
        return headers

    @staticmethod
    def _track_metadata(track: Track) -> dict[str, Any]:
        return {
            "title": track.name,
            "artists": [artist.name for artist in track.artists],
            "album": track.album.name if track.album else None,
            "recording_mbid": track.mbid,
            "isrc": track.get_external_id(ExternalID.ISRC),
        }

    @staticmethod
    def _metadata_complete(metadata: dict[str, Any]) -> bool:
        duration = metadata.get("duration")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not 0 < duration <= MAX_DURATION
        ):
            return False
        if any(
            not isinstance(metadata.get(key), str) or not metadata[key].strip()
            for key in ("title", "album")
        ):
            return False
        artists = metadata.get("artists")
        if not isinstance(artists, list) or not any(
            isinstance(artist, str) and artist.strip() for artist in artists
        ):
            return False
        isrc = metadata.get("isrc")
        if isinstance(isrc, str) and re.fullmatch(r"[A-Za-z]{2}[A-Za-z0-9]{3}[0-9]{7}", isrc):
            return True
        mbid = metadata.get("recording_mbid")
        if isinstance(mbid, str):
            try:
                UUID(mbid)
            except ValueError:
                return False
            return True
        return False

    @classmethod
    def _repair_header(cls, raw: bytes, track: Track | None) -> str | None:
        header = json.loads(raw)
        if not isinstance(header, dict):
            raise TypeError("Invalid analysis header")
        extra = header.get("extra_data")
        if not isinstance(extra, dict):
            return None
        document = extra.get("dejavu")
        if not isinstance(document, dict) or not isinstance(document.get("metadata"), dict):
            return None
        metadata = document["metadata"]
        if cls._metadata_complete(metadata):
            return None
        if track is None:
            return None
        changed = False
        for key, value in cls._track_metadata(track).items():
            existing = metadata.get(key)
            missing = (
                existing is None
                or existing in ("", [])
                or (isinstance(existing, str) and not existing.strip())
            )
            if missing and value:
                metadata[key] = value
                changed = True
        if not changed:
            return None
        return json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @staticmethod
    def _compact_header(raw: str | bytes) -> str | None:
        header = json.loads(raw)
        if not isinstance(header, dict):
            raise TypeError("Invalid analysis header")
        extra = header.get("extra_data")
        if not isinstance(extra, dict) or not isinstance(extra.get("dejavu"), dict):
            return None
        extra["dejavu"] = encode_document(extra["dejavu"])
        return json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False)

    async def _post_document(self, destination: str, body: str, submission_id: str) -> bool:
        """Submit a document and require an acknowledgement of durable server acceptance."""
        headers = await self._upload_headers()
        headers["Idempotency-Key"] = submission_id
        await self._pace_request()
        if self.unloading or self._configured_destination() != destination:
            return False
        async with self.mass.http_session.post(
            destination,
            data=body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=20),
            allow_redirects=False,
        ) as response:
            if response.status == 429 or response.status >= 500:
                raise UploadDeferred(response.status, self._retry_delay(response.headers))
            if response.status not in (200, 202):
                raise InvalidDataError(
                    f"HTTP {response.status}; check recognition server, API key and application token"
                )
            raw_ack = await self._read_response(
                response, 4096, "Fingerprint acknowledgement exceeds the size limit"
            )
            try:
                acknowledgement = json.loads(raw_ack)
            except (ValueError, UnicodeError) as err:
                raise InvalidDataError("Fingerprint server returned invalid JSON") from err
            if (
                not isinstance(acknowledgement, dict)
                or acknowledgement.get("submission_id") != submission_id
                or acknowledgement.get("status") not in ("queued", "processed", "pending_metadata")
            ):
                raise InvalidDataError("Fingerprint server returned an invalid acknowledgement")
            if acknowledgement["status"] == "pending_metadata":
                self.logger.warning(
                    "Fingerprint contribution accepted but awaiting complete metadata; "
                    "it is not available for recognition"
                )
        return True
