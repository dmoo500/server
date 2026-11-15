"""Landmark audio analysis with consent-scoped, recoverable fingerprint delivery."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any

import aiohttp
from music_assistant_fingerprint import (
    FORMAT_ID,
    LEGACY_FORMAT_ID,
    MAX_DOCUMENT_BYTES,
    MAX_DURATION,
    MAX_TRACK_HASHES,
    SAMPLE_RATE,
    WindowSampler,
)
from music_assistant_fingerprint.codec import ENCODING_ID, decode_document, encode_document
from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import ConfigEntryType, ExternalID, MediaType
from music_assistant_models.errors import InvalidDataError
from music_assistant_models.media_items import Track
from yarl import URL

from music_assistant.controllers.streams.audio_analysis_codec import decode_extra_data
from music_assistant.helpers.app_vars import app_var
from music_assistant.models.audio_analysis import AudioAnalysisData, AudioAnalysisError
from music_assistant.models.audio_analysis_provider import AudioAnalysisProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.media_items import AudioFormat
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
        self._next_request_at = 0.0
        self._capability_destination: str | None = None
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
        try:
            sampler = WindowSampler(
                streamdetails.duration,
                audio_format.sample_rate,
                audio_format.channels,
                f"pcm_{audio_format.content_type.value}",
            )
        except ValueError as err:
            raise AudioAnalysisError(str(err)) from err
        metadata: dict[str, Any] = {"duration": streamdetails.duration}
        destination = self._configured_destination()
        if destination:
            # Only existing library/queue metadata is consulted; this never fetches provider
            # audio or triggers online enrichment. Stream URLs and provider IDs are not submitted.
            track: Track | None
            if streamdetails.provider == "library":
                track = await self.mass.music.tracks.get_library_item(streamdetails.item_id)
            else:
                track = await self.mass.music.tracks.get_library_item_by_prov_id(
                    streamdetails.item_id, streamdetails.provider
                )
            if track is None and streamdetails.queue_id:
                offset = 0
                while items := self.mass.player_queues.items(streamdetails.queue_id, 100, offset):
                    track = next(
                        (
                            item.media_item
                            for item in items
                            if item.streamdetails is not None
                            and item.streamdetails.uri == streamdetails.uri
                            and isinstance(item.media_item, Track)
                        ),
                        None,
                    )
                    if track is not None:
                        break
                    offset += len(items)
            if track is not None:
                metadata.update(
                    title=track.name,
                    artists=[artist.name for artist in track.artists],
                    album=track.album.name if track.album else None,
                    recording_mbid=track.mbid,
                    isrc=track.get_external_id(ExternalID.ISRC),
                )
            elif streamdetails.stream_metadata is not None:
                metadata.update(
                    title=streamdetails.stream_metadata.title,
                    artists=[streamdetails.stream_metadata.artist]
                    if streamdetails.stream_metadata.artist
                    else [],
                    album=streamdetails.stream_metadata.album,
                )
            if (
                not metadata.get("title")
                or not metadata.get("artists")
                or not metadata.get("album")
            ):
                self.logger.warning(
                    "Fingerprint metadata lacks title, artist or album; the receiver must "
                    "hold this contribution pending metadata"
                )
        self._data[session_id] = FingerprintSession(sampler, destination, metadata)
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
        return AudioAnalysisData(
            duration=data.metadata["duration"],
            extra_data={"dejavu": document, "upload_destination": data.destination},
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
            async for row in self.mass.streams.audio_analysis.iter_audio_analysis_rows(self.domain):
                try:
                    extra = await self._run_offloaded(decode_extra_data, row["header"])
                except (ValueError, UnicodeError) as err:
                    self.logger.warning("Cannot decode stored fingerprints: %s", err)
                    continue
                if extra is not None:
                    await self._upload(extra)
                if self._upload_blocked or self.unloading or self._retry_at > time.time():
                    return
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

    async def _upload(self, extra: dict[str, Any]) -> None:
        """Deliver one eligible result with idempotency and explicit failure handling."""
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
            try:
                decoded = await self._run_offloaded(decode_document, document)
                body = await self._run_offloaded(
                    json.dumps, decoded, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
            except (ValueError, TypeError, UnicodeError) as err:
                self.logger.error("Invalid stored fingerprint upload: %s", err)
                return
            if len(body.encode("utf-8")) > MAX_RESULT_BYTES:
                self.logger.error("Stored fingerprint upload exceeds the size limit")
                return
            submission_id = hashlib.sha256(body.encode("utf-8")).hexdigest()
            cache_key = hashlib.sha256(f"{destination}|{submission_id}".encode()).hexdigest()
            status = await self.mass.cache.get(cache_key, provider=self.domain, allow_bypass=False)
            if isinstance(status, dict):
                if status.get("accepted") is True:
                    return
                if status.get("retry_at", 0) > time.time():
                    self._retry_at = status["retry_at"]
                    self._schedule_retry()
                    return
            if self.unloading or self._configured_destination() != destination:
                return
            retry_delay = RETRY_SECONDS
            try:
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
            except (ValueError, TypeError, UnicodeError) as err:
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
            await self.mass.cache.set(
                cache_key,
                {"retry_at": self._retry_at},
                provider=self.domain,
                expiration=retry_delay,
            )
            self._schedule_retry()

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
                raise UploadDeferred(response.status, RETRY_SECONDS)
            if response.status != 200:
                raise InvalidDataError(
                    f"Fingerprint capability check returned HTTP {response.status}; "
                    "update the receiver and gateway before enabling compact uploads"
                )
            try:
                raw = await response.content.readexactly(4097)
            except asyncio.IncompleteReadError as err:
                raw = err.partial
            if len(raw) > 4096:
                raise InvalidDataError("Fingerprint capabilities exceed the size limit")
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

    async def _pace_request(self) -> None:
        """Space outbound requests on the serialized upload path at least one second apart."""
        if (delay := self._next_request_at - time.monotonic()) > 0:
            await asyncio.sleep(delay)
        self._next_request_at = time.monotonic() + 1

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
                retry_delay = RETRY_SECONDS
                value = response.headers.get("Retry-After", "")
                if value.isdigit():
                    retry_delay = max(RETRY_SECONDS, min(int(value), 86400))
                elif value:
                    try:
                        seconds = parsedate_to_datetime(value).timestamp() - time.time()
                    except ValueError, TypeError, OverflowError:
                        self.logger.warning("Fingerprint server returned invalid Retry-After")
                    else:
                        retry_delay = max(RETRY_SECONDS, min(math.ceil(seconds), 86400))
                raise UploadDeferred(response.status, retry_delay)
            if response.status not in (200, 202):
                raise InvalidDataError(
                    f"HTTP {response.status}; check recognition server, API key and application token"
                )
            try:
                raw_ack = await response.content.readexactly(4097)
            except asyncio.IncompleteReadError as err:
                raw_ack = err.partial
            if len(raw_ack) > 4096:
                raise InvalidDataError("Fingerprint acknowledgement exceeds the size limit")
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
