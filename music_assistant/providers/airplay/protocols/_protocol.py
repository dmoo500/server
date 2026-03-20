"""Base protocol class for AirPlay streaming implementations."""

from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from music_assistant_models.enums import PlaybackState

from music_assistant.helpers.named_pipe import AsyncNamedPipeWriter
from music_assistant.providers.airplay.constants import AIRPLAY_PCM_FORMAT
from music_assistant.providers.airplay.helpers import generate_active_remote_id

if TYPE_CHECKING:
    from music_assistant_models.player import PlayerMedia

    from music_assistant.helpers.process import AsyncProcess
    from music_assistant.providers.airplay.player import AirPlayPlayer
    from music_assistant.providers.airplay.stream_session import AirPlayStreamSession


class AirPlayProtocol(ABC):
    """Base class for AirPlay streaming protocols (RAOP and AirPlay2).

    This class contains common logic shared between protocol implementations,
    with abstract methods for protocol-specific behavior.
    """

    _cli_proc: AsyncProcess | None  # reference to the (protocol-specific) CLI process
    session: AirPlayStreamSession | None = None  # reference to the active stream session (if any)

    # the pcm audio format used for streaming to this protocol
    pcm_format = AIRPLAY_PCM_FORMAT

    def __init__(
        self,
        player: AirPlayPlayer,
    ) -> None:
        """Initialize base AirPlay protocol.

        Args:
            player: The player to stream to
        """
        self.prov = player.provider
        self.mass = player.provider.mass
        self.player = player
        self.logger = player.provider.logger.getChild(f"protocol.{self.__class__.__name__}")
        mac_address = self.player.device_info.mac_address or self.player.player_id
        self.active_remote_id: str = generate_active_remote_id(mac_address)
        self.prevent_playback: bool = False
        self._cli_proc: AsyncProcess | None = None
        self.commands_pipe = AsyncNamedPipeWriter(
            f"/tmp/{self.player.protocol.value}-{self.player.player_id}-{self.active_remote_id}-cmd",  # noqa: S108
        )
        self._stopped = False
        self._total_bytes_sent = 0
        self._stream_bytes_sent = 0
        self._connected = asyncio.Event()
        self._metadata_checksum = ""
        self._last_metadata_sent: float = 0.0
        self._progress_task: asyncio.Task[None] | None = None
        self._artwork_tmpfile: str | None = None
        self._artwork_url: str | None = None  # last image_url for which artwork was prepared
        self._force_artwork_refresh: bool = False  # trigger two-step SENDMETA on next retry

    @property
    def running(self) -> bool:
        """Return boolean if this stream is running."""
        return not self._stopped and self._cli_proc is not None and not self._cli_proc.closed

    @abstractmethod
    async def start(self, start_ntp: int) -> None:
        """Start the CLI process.

        :param start_ntp: NTP timestamp to start streaming.
        """

    async def wait_for_connection(self) -> None:
        """Wait for device connection to be established."""
        if not self._cli_proc:
            return
        await asyncio.wait_for(self._connected.wait(), timeout=10)
        # repeat sending the volume level to the player because some players seem
        # to ignore it the first time
        # https://github.com/music-assistant/support/issues/3330
        self.mass.call_later(2, self.send_cli_command(f"VOLUME={self.player.volume_level}"))
        # we also need to send the metadata after connection, because some players (e.g. Sonos)
        # simply won't start playback until they receive the metadata ?!
        # Schedule multiple sends to handle Apple TV state transitions after other apps were used:
        # the ATV may ignore the first SENDMETA while releasing a previous app's Now Playing lock.
        self.mass.call_later(2, self.player._on_player_media_updated)
        self.mass.call_later(7, self._retry_metadata)
        self.mass.call_later(15, self._retry_metadata)
        # start periodic progress updates for Now Playing display on the device
        self._progress_task = self.mass.create_task(self._progress_updater())

    async def stop(self, force: bool = False) -> None:
        """
        Stop playback and cleanup.

        :param force: If True, immediately kill the process without graceful shutdown.
        """
        # cancel the periodic progress update task
        if self._progress_task and not self._progress_task.done():
            self._progress_task.cancel()
        # always send stop command first
        await self.send_cli_command("ACTION=STOP")
        self._stopped = True
        await self.commands_pipe.remove()
        # clean up artwork temp file if one was created
        if self._artwork_tmpfile and os.path.exists(self._artwork_tmpfile):
            Path(self._artwork_tmpfile).unlink()
            self._artwork_tmpfile = None
        if force:
            # Kill immediately - skip write_eof() as it can block indefinitely
            # when the CLI stops reading from stdin after receiving STOP.
            if self._cli_proc and not self._cli_proc.closed:
                await self._cli_proc.kill()
        else:
            if self._cli_proc:
                await self._cli_proc.write_eof()
            if self._cli_proc and not self._cli_proc.closed:
                await self._cli_proc.close()
        self.player.set_state_from_stream(state=PlaybackState.IDLE, elapsed_time=0)

    async def write_audio(self, data: bytes) -> None:
        """Write raw audio data to the CLI process stdin.

        :param data: Raw audio bytes to send to the streaming process.
        """
        if self._stopped or not self._cli_proc or self._cli_proc.closed:
            return
        await self._cli_proc.write(data)

    async def write_audio_eof(self) -> None:
        """Signal end-of-stream to the CLI process stdin."""
        if self._stopped or not self._cli_proc or self._cli_proc.closed:
            return
        await self._cli_proc.write_eof()

    async def send_cli_command(self, command: str) -> None:
        """Send an interactive command to the running CLI binary."""
        if self._stopped or not self._cli_proc or self._cli_proc.closed:
            return
        if not self.commands_pipe:
            return
        self.player.last_command_sent = time.time()
        if not command.endswith("\n"):
            command += "\n"
        await self.commands_pipe.write(command.encode("utf-8"))

    def _retry_metadata(self) -> None:
        """Force a metadata re-send by clearing the checksum cache and triggering an update."""
        if self._stopped:
            return
        self.logger.debug(
            "%s: Triggering metadata refresh (Now Playing update)", self.player.display_name
        )
        # Reset checksum so send_metadata does not skip the resend as duplicate.
        # Also set _force_artwork_refresh so send_metadata will first send a SENDMETA
        # without artwork (clearing the ATV's cached artwork), then immediately resend
        # with artwork — forcing the ATV to re-fetch even if the picohttp URL is the same.
        self._metadata_checksum = ""
        self._artwork_url = None
        self._force_artwork_refresh = True
        self.player._on_player_media_updated()

    async def _progress_updater(self) -> None:
        """Periodically send progress and metadata updates to the player for Now Playing display."""
        await asyncio.sleep(5)
        ticks = 0
        while not self._stopped:
            if ticks % 6 == 0:
                # Every 30 seconds force a full metadata re-send (incl. artwork) to
                # recover from state changes: screensaver, app switches on Apple TV.
                # This runs regardless of playback state so it also fires after
                # the ATV returns from screensaver or another app.
                self._retry_metadata()
            elif self.player.playback_state == PlaybackState.PLAYING:
                media = self.player.state.current_media
                if media:
                    progress = int(media.corrected_elapsed_time or 0)
                    await self.send_cli_command(f"PROGRESS={progress}")
            await asyncio.sleep(5)
            ticks += 1

    async def _prepare_artwork(self, image_url: str) -> str:
        """Return the value for the CLI ARTWORK= command.

        Default implementation rewrites MA-internal imageproxy URLs to use a
        loopback HTTP address so cliap2 can always reach the local endpoint,
        regardless of whether MA is behind an HTTPS reverse proxy or
        publish_ip is 0.0.0.0. External CDN URLs are passed through unchanged.
        RaopStream overrides this to download the image directly to a temp file.

        :param image_url: The original image URL from PlayerMedia.
        """
        if "/imageproxy?" in image_url:
            # Rewrite MA-internal imageproxy URL to guaranteed loopback HTTP
            local_base = f"http://127.0.0.1:{self.mass.streams.publish_port}"
            for ma_base in (self.mass.streams.base_url, self.mass.webserver.base_url):
                if image_url.startswith(ma_base):
                    return local_base + image_url[len(ma_base) :]
        return image_url

    async def send_metadata(self, progress: int | None, metadata: PlayerMedia | None) -> None:
        """Send metadata to player."""
        if self._stopped:
            return
        if metadata:
            duration = min(metadata.duration or 0, 3600)
            title = metadata.title or ""
            artist = metadata.artist or ""
            album = metadata.album or ""

            metadata_checksum = f"{title}|{artist}|{album}|{duration}|{metadata.image_url}"
            if (
                metadata_checksum == self._metadata_checksum
                and time.time() - self._last_metadata_sent <= 2
            ):
                # metadata has not changed since last time, skip sending to CLI
                return
            self._metadata_checksum = metadata_checksum
            self._last_metadata_sent = time.time()

            # Send metadata fields as separate pipe writes to avoid truncation on
            # non-blocking FIFO writes when payloads become large.
            async def _send_base_metadata_fields() -> None:
                await self.send_cli_command(f"TITLE={title}")
                await self.send_cli_command(f"ARTIST={artist}")
                await self.send_cli_command(f"ALBUM={album}")
                await self.send_cli_command(f"DURATION={duration}")

            if metadata.image_url:
                artwork_value = await self._prepare_artwork(metadata.image_url)
                self.logger.debug(
                    "%s: Sending metadata — title=%r artist=%r album=%r artwork=%s",
                    self.player.display_name,
                    title,
                    artist,
                    album,
                    artwork_value,
                )
                if self._force_artwork_refresh:
                    # Two-step SENDMETA: first without artwork to clear the ATV's cached
                    # artwork entry, then immediately with artwork to force a re-fetch.
                    # This is necessary because the ATV caches artwork by picohttp URL;
                    # sending the same URL again after a screensaver/app-switch is ignored.
                    self._force_artwork_refresh = False
                    await _send_base_metadata_fields()
                    await self.send_cli_command("PROGRESS=0")
                    await self.send_cli_command("ACTION=SENDMETA")
                    await asyncio.sleep(0.5)
                await _send_base_metadata_fields()
                await self.send_cli_command(f"ARTWORK={artwork_value}")
            else:
                self.logger.debug(
                    "%s: Sending metadata — title=%r artist=%r album=%r (no artwork)",
                    self.player.display_name,
                    title,
                    artist,
                    album,
                )
                self._force_artwork_refresh = False
                await _send_base_metadata_fields()

            await self.send_cli_command("PROGRESS=0")
            await self.send_cli_command("ACTION=SENDMETA")
        if progress is not None:
            await self.send_cli_command(f"PROGRESS={progress}")
