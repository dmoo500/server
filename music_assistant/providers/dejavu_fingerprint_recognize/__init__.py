"""Recognize selected live inputs using local acoustic fingerprints."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import aiohttp
from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import ConfigEntryType, MediaType, ProviderFeature
from music_assistant_models.errors import InvalidDataError
from music_assistant_models.streamdetails import StreamMetadata
from yarl import URL

from music_assistant.helpers.throttle_retry import parse_retry_after
from music_assistant.models.audio_analysis_provider import AudioAnalysisProvider
from music_assistant.models.media_capabilities import AudioSourceMixin

from .extract import extract

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.media_items import AudioFormat
    from music_assistant_models.provider import ProviderManifest
    from music_assistant_models.streamdetails import StreamDetails

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

CAPTURE_LENGTHS = (3, 5, 10, 15)
MATCH_INTERVAL = 30
MAX_MATCH_INTERVAL = 60
RETRY_INTERVAL = 5
MAX_RESPONSE_BYTES = 65536
MAX_CAPTURE_BYTES = 16 * 1024 * 1024


@dataclass
class LiveSession:
    """Bounded state for one selected live input."""

    sample_rate: int
    channels: int
    encoding: str
    frame_bytes: int
    placeholder: StreamMetadata
    buffer: bytearray = field(default_factory=bytearray)
    elapsed_bytes: int = 0
    last_capture: int = 0
    capture_index: int = 0
    interval: float = RETRY_INTERVAL
    task: asyncio.Task[None] | None = None
    stopped: bool = False


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize the live fingerprint recognizer."""
    return FingerprintRecognizer(mass, manifest, config)


def _parse_metadata(
    body: bytes, placeholder: StreamMetadata, capture_seconds: float, captured_at: float
) -> tuple[bool, StreamMetadata, float | None]:
    result = json.loads(body)
    if not isinstance(result, dict):
        raise TypeError("Invalid fingerprint response")
    status = result.get("status")
    if status in ("no_match", "ambiguous") and result.get("match") is None:
        return False, placeholder, None
    if status != "matched" or not isinstance(match := result.get("match"), dict):
        raise ValueError("Invalid fingerprint status")
    data = match.get("metadata")
    if not isinstance(data, dict) or not isinstance(data.get("title"), str):
        raise TypeError("Invalid fingerprint match metadata")
    artists = data.get("artists")
    if not isinstance(artists, list) or not all(isinstance(item, str) for item in artists):
        raise ValueError("Invalid fingerprint artist credits")
    album = data.get("album")
    if album is not None and not isinstance(album, str):
        raise ValueError("Invalid fingerprint album")
    metadata = StreamMetadata(title=data["title"], artist=", ".join(artists) or None, album=album)
    duration, offset = data.get("duration"), match.get("offset_seconds")
    if (
        isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and isinstance(offset, (int, float))
        and not isinstance(offset, bool)
        and 1 <= duration <= 1800
        and offset >= -0.1
        and offset + capture_seconds <= duration + 3
    ):
        position = min(duration, max(0.0, offset + capture_seconds))
        metadata.duration = round(duration)
        metadata.elapsed_time = int(position)
        metadata.elapsed_time_last_updated = captured_at
        return (
            True,
            metadata,
            max(RETRY_INTERVAL, min(MAX_MATCH_INTERVAL, duration - position + CAPTURE_LENGTHS[0])),
        )
    return True, metadata, None


async def _read_response(response: aiohttp.ClientResponse) -> bytes:
    body = bytearray()
    async for chunk in response.content.iter_chunked(8192):
        body.extend(chunk)
        if len(body) > MAX_RESPONSE_BYTES:
            raise ValueError("Fingerprint response exceeds the size limit")
    return bytes(body)


class FingerprintRecognizer(AudioAnalysisProvider):
    """Identify audio sources without retaining or transmitting audio."""

    media_types = frozenset({MediaType.AUDIO_SOURCE})

    async def handle_async_init(self) -> None:
        """Validate the recognition server and credentials."""
        self._live: dict[str, LiveSession] = {}
        self._request_lock = asyncio.Lock()
        self._next_request = 0.0
        self._disabled = False
        value = self.get_setup_value("server_url")
        if not isinstance(value, str) or not value:
            raise InvalidDataError("A fingerprint server URL is required")
        url = URL(value)
        local_http = url.scheme == "http" and url.host in ("localhost", "127.0.0.1", "::1")
        if (
            not url.host
            or (url.scheme != "https" and not local_http)
            or url.user is not None
            or url.query_string
            or url.fragment
        ):
            raise InvalidDataError("Use HTTPS without URL credentials, query or fragment")
        self._url = str(url).rstrip("/") + "/api/v1/dejavu/recognize/landmarks"
        key = self.get_setup_value("api_key")
        if not isinstance(key, str) or not key.strip():
            raise InvalidDataError("A fingerprint recognition API key is required")
        self._headers = {"X-API-Key": key}
        application_key = self.get_setup_value("application_key")
        if isinstance(application_key, str) and application_key:
            self._headers["X-Application-Key"] = application_key

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return server credentials and explicitly selected live sources."""
        options: list[ConfigValueOption] = []
        for provider in self.mass.providers:
            if not isinstance(provider, AudioSourceMixin) or (
                ProviderFeature.AUDIO_SOURCE not in provider.supported_features
            ):
                continue
            sources = await provider.get_audio_sources()
            options.extend(
                ConfigValueOption(
                    f"{provider.instance_id}|{source.item_id}",
                    title=f"{provider.name}: {source.name}",
                )
                for source in sources
            )
        return (
            ConfigEntry(
                key="sources",
                type=ConfigEntryType.STRING,
                multi_value=True,
                required=False,
                default_value=[],
                options=options,
            ),
        )

    async def process_pcm_chunk(self, session_id: str, pcm_chunk: bytes) -> None:
        """Copy a bounded capture without blocking the live stream."""
        session = self._live.get(session_id)
        if session is None or session.stopped or self._disabled:
            return
        if len(pcm_chunk) % session.frame_bytes:
            self.logger.warning("Invalid PCM frame alignment for live recognition")
            return
        if len(pcm_chunk) > 2 * session.sample_rate * session.frame_bytes:
            self.logger.warning("Live recognition received an oversized PCM chunk")
            return
        session.elapsed_bytes += len(pcm_chunk)
        session.buffer.extend(pcm_chunk)
        max_bytes = CAPTURE_LENGTHS[-1] * session.sample_rate * session.frame_bytes
        if len(session.buffer) > max_bytes:
            del session.buffer[: len(session.buffer) - max_bytes]
        capture_bytes = (
            CAPTURE_LENGTHS[session.capture_index] * session.sample_rate * session.frame_bytes
        )
        interval = session.interval * session.sample_rate * session.frame_bytes
        if len(session.buffer) < capture_bytes or (session.task and not session.task.done()):
            return
        if session.last_capture and session.elapsed_bytes - session.last_capture < interval:
            return
        if time.monotonic() < self._next_request:
            return
        previous_capture = session.last_capture
        session.last_capture = session.elapsed_bytes
        session.task = self.mass.create_task(
            self._recognize(
                session_id,
                session,
                bytes(session.buffer[-capture_bytes:]),
                time.time(),
                previous_capture,
            )
        )

    async def cancel(self, session_id: str) -> None:
        """Cancel recognition and release the input capture."""
        await self._close_session(session_id)
        await super().cancel(session_id)

    async def _start_analysis(
        self, session_id: str, streamdetails: StreamDetails, audio_format: AudioFormat
    ) -> bool:
        """Accept only a selected source with a supported PCM format."""
        selected = self.config.get_value("sources") or []
        if isinstance(selected, str):
            selected = [selected]
        if not isinstance(selected, list):
            self.logger.warning("Invalid live source selection for fingerprint recognition")
            return False
        if f"{streamdetails.provider}|{streamdetails.item_id}" not in selected:
            return False
        encoding = f"pcm_{audio_format.content_type.value}"
        widths = {"pcm_s16le": 2, "pcm_s24le": 3, "pcm_s32le": 4, "pcm_f32le": 4, "pcm_f64le": 8}
        if (
            encoding not in widths
            or not 1 <= audio_format.channels <= 8
            or not (8000 <= audio_format.sample_rate <= 192000)
            or (
                CAPTURE_LENGTHS[-1]
                * audio_format.sample_rate
                * audio_format.channels
                * widths[encoding]
                > MAX_CAPTURE_BYTES
            )
        ):
            self.logger.warning("Unsupported live input format for fingerprint recognition")
            return False
        self._live[session_id] = LiveSession(
            sample_rate=audio_format.sample_rate,
            channels=audio_format.channels,
            encoding=encoding,
            frame_bytes=widths[encoding] * audio_format.channels,
            placeholder=streamdetails.stream_metadata or StreamMetadata(title="Live input"),
        )
        return True

    async def _finalize(self, session_id: str) -> None:
        """Release a completed live input without storing analysis data."""
        await self._close_session(session_id)

    async def _close_session(self, session_id: str) -> None:
        session = self._live.pop(session_id, None)
        if session and session.task:
            session.task.cancel()
            await asyncio.gather(session.task, return_exceptions=True)

    async def _recognize(
        self,
        session_id: str,
        session: LiveSession,
        pcm: bytes,
        captured_at: float,
        previous_capture: int,
    ) -> None:
        try:
            payload = await self._run_offloaded(
                extract, pcm, session.sample_rate, session.channels, session.encoding
            )
            if not payload["fingerprints"]:
                if not self.report_stream_metadata(session_id, session.placeholder):
                    session.stopped = True
                    session.buffer.clear()
                session.capture_index = 0
                session.interval = MATCH_INTERVAL
                return
            async with self._request_lock:
                if self._disabled:
                    return
                if time.monotonic() < self._next_request:
                    session.last_capture = previous_capture
                    return
                self._next_request = time.monotonic() + 2.1
                async with self.mass.http_session.post(
                    self._url,
                    json=payload,
                    headers=self._headers,
                    timeout=aiohttp.ClientTimeout(total=20),
                    allow_redirects=False,
                ) as response:
                    if response.status == 429 or response.status >= 500:
                        self._next_request = max(
                            self._next_request,
                            time.monotonic()
                            + max(60, parse_retry_after(response.headers.get("Retry-After"))),
                        )
                        if response.status >= 500:
                            session.capture_index = 0
                        self.logger.warning(
                            "Fingerprint recognition deferred: HTTP %s", response.status
                        )
                        return
                    if response.status in (401, 403):
                        self._disabled = True
                    response.raise_for_status()
                    body = await _read_response(response)
            matched, metadata, next_interval = _parse_metadata(
                body,
                session.placeholder,
                len(pcm) / (session.frame_bytes * session.sample_rate),
                captured_at,
            )
            if matched:
                session.capture_index = 0
                session.interval = next_interval if next_interval is not None else MATCH_INTERVAL
            else:
                was_longest = session.capture_index == len(CAPTURE_LENGTHS) - 1
                session.capture_index = min(session.capture_index + 1, len(CAPTURE_LENGTHS) - 1)
                session.interval = MATCH_INTERVAL if was_longest else RETRY_INTERVAL
            if not self.report_stream_metadata(session_id, metadata):
                session.stopped = True
                session.buffer.clear()
        except (aiohttp.ClientError, TimeoutError, ValueError, TypeError) as err:
            self.logger.warning("Fingerprint recognition failed: %s", err)
