"""Logic for RAOP audio streaming to AirPlay devices."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

from music_assistant_models.enums import PlaybackState

from music_assistant.constants import VERBOSE_LOG_LEVEL
from music_assistant.helpers.process import AsyncProcess
from music_assistant.providers.airplay.constants import (
    CONF_ALAC_ENCODE,
    CONF_ENCRYPTION,
    CONF_PASSWORD,
    CONF_RAOP_CREDENTIALS,
)
from music_assistant.providers.airplay.helpers import get_cli_binary

from ._protocol import AirPlayProtocol

if TYPE_CHECKING:
    from music_assistant.providers.airplay.provider import AirPlayProvider


class RaopStream(AirPlayProtocol):
    """
    RAOP (AirPlay 1) Audio Streamer.

    Python is not suitable for realtime audio streaming so we do the actual streaming
    of (RAOP) audio using a small executable written in C based on libraop to do
    the actual timestamped playback, which reads pcm audio from stdin
    and we can send some interactive commands using a named pipe.
    """

    async def start(self, start_ntp: int) -> None:
        """Start CLIRaop process."""
        if self.player.raop_discovery_info is None:
            raise RuntimeError(f"RAOP service not discovered for {self.player.display_name}")
        cli_binary = await get_cli_binary(self.player.protocol)
        extra_args: list[str] = []
        if_ip = self._resolve_if_ip()
        extra_args += ["-if", if_ip]
        if self.player.config.get_value(CONF_ENCRYPTION, True):
            extra_args += ["-encrypt"]
        if self.player.config.get_value(CONF_ALAC_ENCODE, True):
            extra_args += ["-alac"]
        for prop in ("et", "md", "am", "pk", "pw"):
            if prop_value := self.player.raop_discovery_info.decoded_properties.get(prop):
                extra_args += [f"-{prop}", prop_value]
        if device_password := self.player.config.get_value(CONF_PASSWORD):
            extra_args += ["-password", str(device_password)]
        # Add RAOP credentials from pairing if available (for Apple devices)
        if raop_credentials := self.player.config.get_value(CONF_RAOP_CREDENTIALS):
            # Credentials format is "client_id:auth_secret", cliraop expects just auth_secret
            creds_str = str(raop_credentials)
            auth_secret = creds_str.split(":", 1)[1] if ":" in creds_str else creds_str
            extra_args += ["-secret", auth_secret]
        if self.prov.logger.isEnabledFor(logging.DEBUG):
            extra_args += ["-debug", "5"]
        elif self.prov.logger.isEnabledFor(VERBOSE_LOG_LEVEL):
            extra_args += ["-debug", "10"]

        cliraop_args = [
            cli_binary,
            "-ntpstart",
            str(start_ntp),
            "-port",
            str(self.player.raop_discovery_info.port),
            "-latency",
            str(self.player.output_buffer_duration_ms),
            "-volume",
            str(self.player.volume_level),
            *extra_args,
            "-dacp",
            cast("AirPlayProvider", self.prov).dacp_id,
            "-activeremote",
            self.active_remote_id,
            "-cmdpipe",
            self.commands_pipe.path,
            "-udn",
            self.player.raop_discovery_info.name,
            self.player.address,
            "-",  # Use stdin for audio input
        ]
        self.player.logger.debug(
            "Starting cliraop process for player %s with args: %s",
            self.player.player_id,
            cliraop_args,
        )
        self._cli_proc = AsyncProcess(cliraop_args, stdin=True, stderr=True, name="cliraop")
        await self._cli_proc.start()
        # start reading the stderr of the cliap2 process from another task
        self._cli_proc.attach_stderr_reader(self.mass.create_task(self._stderr_reader()))

    def _resolve_if_ip(self) -> str:
        """Resolve best local interface IP for cliraop's -if argument."""
        # Determine the correct local interface IP to pass as -if to cliraop, so
        # that cliraop advertises the right source IP in SDP and the Apple TV can
        # route UDP timing/control packets back to us.
        # Strategy:
        # 1. Use the OS routing table via a connected UDP socket to find the outbound
        #    interface for the specific Apple TV address. This correctly handles
        #    multi-homed hosts where different Apple TVs are on different subnets.
        # 2. Fall back to publish_ip if it is a bindable local interface (handles
        #    Docker/OrbStack where the container needs to advertise the host IP).
        # 3. Fall back to bind_ip as a last resort.
        target_ip = str(self.player.device_info.ip_address)
        if_ip = str(self.mass.streams.bind_ip)
        _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            _s.connect((target_ip, 80))
            routed_ip = _s.getsockname()[0]
            if routed_ip and routed_ip not in ("0.0.0.0", ""):
                if_ip = routed_ip
        except OSError:
            pass
        finally:
            _s.close()
        if if_ip in ("0.0.0.0", ""):
            # Routing gave no usable result; try publish_ip as a bindable override
            # (Docker/OrbStack scenario where the host IP should be advertised)
            publish_ip = str(self.mass.streams.publish_ip or "")
            if publish_ip:
                _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    _s.bind((publish_ip, 0))
                    if_ip = publish_ip
                except OSError:
                    pass
                finally:
                    _s.close()
        return if_ip

    async def _stderr_reader(self) -> None:
        """Monitor stderr for the running CLIRaop process."""
        player = self.player
        logger = player.logger
        lost_packets = 0
        if not self._cli_proc:
            return
        async for line in self._cli_proc.iter_stderr():
            if self._stopped:
                break
            if "connected to " in line:
                self._connected.set()
                # successfully connected - playback will/can start
            if "set pause" in line or "Pause at" in line:
                player.set_state_from_stream(state=PlaybackState.PAUSED, stream=self)
            elif "Restarted at" in line or "restarting w/ pause" in line:
                player.set_state_from_stream(state=PlaybackState.PLAYING, stream=self)
            elif "restarting w/o pause" in line:
                # streaming has started - send metadata now so cliraop has it
                # available immediately during this RTP restart window, which is
                # when the Apple TV is most likely to accept a Now Playing update.
                player.set_state_from_stream(
                    state=PlaybackState.PLAYING, elapsed_time=0, stream=self
                )
                self._retry_metadata()
            elif "elapsed milliseconds:" in line:
                # this is received more or less every second while playing
                millis = int(line.split("elapsed milliseconds: ")[1])
                # note that this represents the total elapsed time of the streaming session
                elapsed_time = millis / 1000
                player.set_state_from_stream(elapsed_time=elapsed_time)
            elif "Password required, but none supplied." in line:
                logger.error(
                    f"Player {self.player.name} requires a password. "
                    f"Please add one in Player Settings"
                )
                break
            if "lost packet out of backlog" in line:
                lost_packets += 1
                if lost_packets == 100:
                    logger.error("High packet loss detected, restarting playback...")
                    self.mass.create_task(self.mass.players.cmd_resume(self.player.player_id))
                else:
                    logger.warning("Packet loss detected!")
            if "end of stream reached" in line:
                logger.debug("End of stream reached")
                break
            if "Error opening input" in line or "Error opening input file" in line:
                # ffmpeg error when binary tries to process a missing artwork image;
                # this is a non-fatal binary-side issue, log at debug level only
                logger.debug("Artwork processing error (no image available): %s", line)
            elif (
                "artwork" in line.lower()
                or "metadata" in line.lower()
                or "pico" in line.lower()
                or "socket" in line.lower()
                or "http server" in line.lower()
            ):
                # Artwork/metadata/picohttp status lines — log at DEBUG to diagnose
                # whether the ATV fetches the image from cliraop's picohttp server.
                logger.debug("[cliraop] %s", line.strip())
            else:
                logger.log(VERBOSE_LOG_LEVEL, line)
            await asyncio.sleep(0)  # Yield to event loop

        logger.debug("CLIRaop stderr reader ended")
        if not self._stopped:
            self._stopped = True
            self.player.set_state_from_stream(state=PlaybackState.IDLE, elapsed_time=0, stream=self)

    async def _prepare_artwork(self, image_url: str) -> str:
        """Download artwork to a local temp file and return its file path for cliraop.

        cliraop receives the ARTWORK= value via a named pipe which has a limited
        line-buffer size. Long imageproxy or CDN URLs (with encoded query parameters)
        get silently truncated, causing a 400 Bad Request when cliraop tries to
        download them. Passing a short local file path avoids this entirely.

        The downloaded file is reused across metadata retries as long as the image
        URL hasn't changed (i.e. the same track is still playing).

        :param image_url: The original image URL from PlayerMedia.
        """
        # Reuse the existing temp file if the image URL hasn't changed
        if (
            image_url == self._artwork_url
            and self._artwork_tmpfile
            and os.path.exists(self._artwork_tmpfile)
        ):
            return self._artwork_tmpfile

        try:
            # For MA-internal imageproxy URLs, rewrite to guaranteed loopback HTTP
            fetch_url = image_url
            if "/imageproxy?" in image_url:
                local_base = f"http://127.0.0.1:{self.mass.streams.publish_port}"
                for ma_base in (self.mass.streams.base_url, self.mass.webserver.base_url):
                    if image_url.startswith(ma_base):
                        fetch_url = local_base + image_url[len(ma_base) :]
                        break

            async with self.mass.http_session.get(fetch_url) as resp:
                if resp.status != 200:
                    self.logger.debug(
                        "%s: Artwork download returned HTTP %s, falling back to URL",
                        self.player.display_name,
                        resp.status,
                    )
                    return image_url
                content_type = resp.headers.get("Content-Type", "image/jpeg")
                image_data = await resp.read()

            ext = ".jpg" if "png" not in content_type else ".png"
            if self._artwork_tmpfile and os.path.exists(self._artwork_tmpfile):
                Path(self._artwork_tmpfile).unlink()
            fd, path = tempfile.mkstemp(
                suffix=ext,
                prefix=f"ma-raop-{self.player.player_id[:8]}-",
            )
            os.write(fd, image_data)
            os.close(fd)
            self._artwork_tmpfile = path
            self._artwork_url = image_url
            self.logger.debug(
                "%s: Artwork saved to temp file %s (%d bytes)",
                self.player.display_name,
                path,
                len(image_data),
            )
            return path
        except Exception as err:
            self.logger.debug(
                "%s: Failed to prepare artwork (%s), falling back to URL",
                self.player.display_name,
                err,
            )
            return image_url
