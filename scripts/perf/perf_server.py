"""
Hermetic Music Assistant server for performance benchmarking.

Boots a real MusicAssistant instance that is fully isolated from the host LAN,
sized via the fake `test` music provider and `_demo_player_provider` players.

Safety properties (do NOT weaken these - see scripts/perf/README.md):

- webserver and streamserver bind/publish on 127.0.0.1 only
- zeroconf/SSDP discovery is mocked: nothing is advertised on or discovered from the LAN
- the sendspin builtin provider is stripped BEFORE builtin load: it runs its own
  aiosendspin server with its own mDNS advertisement (bypassing the mocked discovery
  controller), so real devices on the LAN would otherwise connect to the test instance
- the default device providers are suppressed and only an allowlist of local-only
  builtin providers is loaded, so no host hardware is bridged into the player registry
  and no network metadata providers (musicbrainz, fanarttv, ...) are contacted

Exposes a loopback control channel (aiohttp, --control-port) used by run_benchmark.py:

  GET  /ping                -> liveness/readiness probe
  GET  /sync_active         -> whether any provider sync task is pending/running
  POST /yappi/start         -> start CPU profiling (cpu clock)
  POST /yappi/stop          -> stop profiling, return top rows (by tsub), clear stats
  POST /lag/reset           -> reset the event-loop lag window
  GET  /lag                 -> max event-loop lag (ms) since last reset
  POST /resolve_stream_url  -> resolve the stream URL for a player's current media
    POST /statistics/query_benchmark -> seed synthetic history and measure query latency
  POST /gc                  -> run a full garbage collection (for stable RSS checkpoints)

Set PERF_YAPPI_BOOT=1 to start yappi before the music_assistant import, so the first
/yappi/stop returns import+boot attribution.

An admin user + long-lived token is created on first boot and written to
<data-dir>/token.txt for the websocket bench client.

Run with the repo venv python:
  .venv/bin/python scripts/perf/perf_server.py --data-dir /tmp/xxx --port 8098 --control-port 8099
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import logging
import os
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, NonCallableMagicMock, patch

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant

# ruff: noqa: T201

try:
    import psutil
    import yappi
    from aiohttp import web
except ImportError as err:
    print(
        f"Missing benchmark dependency: {err}.\n"
        "Install the dev/test extras first: uv pip install -e '.[test]'",
        file=sys.stderr,
    )
    sys.exit(1)

LOGGER = logging.getLogger("perf_server")

REPO_ROOT = Path(__file__).resolve().parents[2]

# Local-only builtin providers that are allowed to load. Everything else with
# builtin=true in its manifest (sendspin, musicbrainz, fanarttv, ...)
# is stripped before builtin load: hermetic-by-default, so a future builtin can
# never silently punch a hole in the isolation of the benchmark instance.
ALLOWED_BUILTIN_PROVIDERS = {
    "builtin",
    "loudness_analysis",
    "playlist_metadata",
    "radio_playlist",
    "sync_group",
}

YAPPI_TOP_LIMIT = 15


class LoopLagMonitor:
    """Track the maximum event-loop scheduling latency since the last reset."""

    def __init__(self) -> None:
        """Initialize the monitor."""
        self.max_lag: float = 0.0

    def reset(self) -> None:
        """Reset the tracked maximum lag."""
        self.max_lag = 0.0

    async def run(self) -> None:
        """Continuously sample the event-loop scheduling latency."""
        loop = asyncio.get_running_loop()
        while True:
            t_start = loop.time()
            await asyncio.sleep(0.1)
            lag = loop.time() - t_start - 0.1
            self.max_lag = max(self.max_lag, lag)


class ControlApi:
    """Loopback HTTP control channel for the benchmark orchestrator."""

    def __init__(self, mass: MusicAssistant, lag_monitor: LoopLagMonitor) -> None:
        """
        Initialize the control API.

        :param mass: The running MusicAssistant instance.
        :param lag_monitor: The event-loop lag monitor to expose.
        """
        self.mass = mass
        self.lag_monitor = lag_monitor
        self._runner: web.AppRunner | None = None

    async def start(self, port: int) -> None:
        """Start the control HTTP server on the loopback interface."""
        app = web.Application()
        app.router.add_get("/ping", self._handle_ping)
        app.router.add_get("/sync_active", self._handle_sync_active)
        app.router.add_post("/yappi/start", self._handle_yappi_start)
        app.router.add_post("/yappi/stop", self._handle_yappi_stop)
        app.router.add_post("/lag/reset", self._handle_lag_reset)
        app.router.add_get("/lag", self._handle_lag)
        app.router.add_post("/resolve_stream_url", self._handle_resolve_stream_url)
        app.router.add_post("/statistics/query_benchmark", self._handle_statistics_query_benchmark)
        app.router.add_get("/fingerprints", self._handle_fingerprints)
        app.router.add_post("/fingerprints/setup", self._handle_fingerprint_setup)
        app.router.add_post("/gc", self._handle_gc)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", port)
        await site.start()

    async def stop(self) -> None:
        """Stop the control HTTP server."""
        if self._runner:
            await self._runner.cleanup()

    async def _handle_ping(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def _handle_sync_active(self, request: web.Request) -> web.Response:
        return web.json_response({"active": bool(self.mass.music.active_sync_tasks)})

    async def _handle_yappi_start(self, request: web.Request) -> web.Response:
        yappi.set_clock_type("cpu")
        yappi.start(builtins=False)
        return web.json_response({"status": "started"})

    async def _handle_yappi_stop(self, request: web.Request) -> web.Response:
        yappi.stop()
        rows = _yappi_top_rows()
        yappi.clear_stats()
        return web.json_response({"rows": rows})

    async def _handle_lag_reset(self, request: web.Request) -> web.Response:
        self.lag_monitor.reset()
        return web.json_response({"status": "ok"})

    async def _handle_lag(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "max_lag_ms": round(self.lag_monitor.max_lag * 1000, 2),
                "playback_active": self.mass.streams.audio_analysis.playback_active(),
            }
        )

    async def _handle_gc(self, request: web.Request) -> web.Response:
        collected = gc.collect()
        return web.json_response({"collected": collected})

    async def _handle_resolve_stream_url(self, request: web.Request) -> web.Response:
        body = await request.json()
        # the stream is resolved for the protocol that renders the audio of the player
        protocol = self.mass.players.get_output_protocol(body["player_id"])
        if protocol is None or protocol.current_media is None:
            return web.json_response({"error": "player not found or no current media"}, status=404)
        url = await self.mass.streams.resolve_stream_url(
            protocol.protocol_id, protocol.current_media
        )
        return web.json_response({"url": url})

    async def _handle_statistics_query_benchmark(self, request: web.Request) -> web.Response:
        from statistics_query_benchmark import benchmark_statistics_queries  # noqa: PLC0415

        try:
            result = await benchmark_statistics_queries(self.mass, await request.json())
        except Exception as err:
            LOGGER.exception("Statistics query benchmark failed")
            raise web.HTTPInternalServerError(text=f"{type(err).__name__}: {err}") from err
        return web.json_response(result)

    async def _handle_fingerprints(self, request: web.Request) -> web.Response:
        from audio_fingerprint.codec import decode_document  # noqa: PLC0415

        from music_assistant.controllers.streams.audio_analysis_codec import (  # noqa: PLC0415
            decode_extra_data,
        )

        controller = self.mass.streams.audio_analysis
        results = []
        async for row in controller.iter_audio_analysis_rows("dejavu_analysis"):
            extra = await asyncio.to_thread(decode_extra_data, row["header"])
            if extra is None:
                raise web.HTTPInternalServerError(text="Invalid stored fingerprint document")
            doc = await asyncio.to_thread(decode_document, extra["dejavu"])
            end = 0
            full = True
            for window in doc.get("windows", []):
                if window["start_sample"] > end:
                    full = False
                end = max(end, window["start_sample"] + window["sample_count"])
            results.append(
                {
                    "format_id": doc.get("format_id"),
                    "landmarks": len(doc.get("fingerprints", [])),
                    "duration_seconds": doc.get("metadata", {}).get("duration"),
                    "full": full and end == 60 * 44100,
                    "analysis_version": row["analysis_version"],
                    "item_id": row["item_id"],
                }
            )
        return web.json_response(
            {"results": results, "playback_active": controller.playback_active()}
        )

    async def _handle_fingerprint_setup(self, request: web.Request) -> web.Response:
        """Configure only local benchmark providers with their final fixture sizes."""
        body = await request.json()
        await self.mass.config._create_provider_instance(
            "test",
            {
                "num_artists": 1,
                "num_albums": 1,
                "num_tracks": 6,
                "num_podcasts": 0,
                "num_audiobooks": 0,
            },
        )
        await self.mass.config._create_provider_instance(
            "_demo_player_provider", {"number_of_players": 2}
        )
        if body["enabled"]:
            await self.mass.config._create_provider_instance(
                "dejavu_analysis", {"upload_enabled": False}
            )
        return web.json_response({"status": "configured"})


def prepare_fingerprint_fixture(data_dir: Path) -> Path:
    """Write a deterministic synthetic WAV exclusively in the temporary benchmark directory."""
    import wave  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    target = data_dir / "fingerprint-fixture.wav"
    rng = np.random.default_rng(20261114)
    with wave.open(str(target), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(44100)
        for _ in range(60):
            mono = np.clip(rng.normal(0, 0.1, 44100), -1, 1)
            pcm = np.repeat((mono * 32767).astype("<i2")[:, None], 2, axis=1)
            output.writeframes(pcm.tobytes())
    return target


async def install_fingerprint_fixture(data_dir: Path) -> None:
    """Replace only the hermetic test provider's silence source with the local fixture."""
    from music_assistant_models.enums import ContentType, MediaType, StreamType  # noqa: PLC0415
    from music_assistant_models.media_items import AudioFormat  # noqa: PLC0415
    from music_assistant_models.streamdetails import StreamDetails  # noqa: PLC0415

    from music_assistant.providers.test import TestProvider  # noqa: PLC0415

    fixture = await asyncio.to_thread(prepare_fingerprint_fixture, data_dir)

    async def local_stream_details(
        self: TestProvider, item_id: str, media_type: MediaType
    ) -> StreamDetails:
        return StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            audio_format=AudioFormat(
                content_type=ContentType.WAV, sample_rate=44100, bit_depth=16, channels=2
            ),
            media_type=media_type,
            stream_type=StreamType.LOCAL_FILE,
            path=str(fixture),
            duration=60,
            can_seek=True,
            allow_seek=True,
        )

    TestProvider.get_stream_details = local_stream_details  # type: ignore[method-assign]


def _yappi_top_rows(limit: int = YAPPI_TOP_LIMIT) -> list[dict[str, Any]]:
    """Return the top yappi function stats (by self-time) as structured records."""
    stats = yappi.get_func_stats()
    stats.sort("tsub", "desc")
    rows: list[dict[str, Any]] = []
    for stat in stats:
        # skip the harness' own frames - they are measurement scaffolding
        if "scripts/perf/" in stat.module.replace(os.sep, "/"):
            continue
        rows.append(
            {
                "name": f"{_shorten_module(stat.module)}:{stat.lineno} {stat.name}",
                "ncall": stat.ncall,
                "tsub_ms": round(stat.tsub * 1000, 1),
                "ttot_ms": round(stat.ttot * 1000, 1),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _shorten_module(module: str) -> str:
    """Strip machine-specific path prefixes so profile rows compare across machines."""
    normalized = module.replace(os.sep, "/")
    if "/site-packages/" in normalized:
        return normalized.split("/site-packages/")[-1]
    if match := re.search(r"/lib/python\d+\.\d+/(.*)", normalized):
        return match.group(1)
    try:
        return Path(module).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError, OSError:
        return normalized


def _create_mock_zeroconf() -> MagicMock:
    """Create a mock AsyncZeroconf that prevents real mDNS network I/O."""
    from zeroconf.asyncio import AsyncZeroconf  # noqa: PLC0415

    mock_zc = MagicMock(spec=AsyncZeroconf)
    mock_inner_zc = NonCallableMagicMock()
    mock_inner_zc.cache = NonCallableMagicMock()
    mock_inner_zc.cache.cache = {}  # empty cache - no discovered services
    mock_zc.zeroconf = mock_inner_zc
    mock_zc.async_register_service = AsyncMock()
    mock_zc.async_update_service = AsyncMock()
    mock_zc.async_unregister_service = AsyncMock()
    mock_zc.async_close = AsyncMock()
    return mock_zc


def allocation_tracking(path: Path | None) -> Any:
    """Return an optional native-allocation tracker for an isolated profiling run."""
    from contextlib import nullcontext  # noqa: PLC0415

    if path is None:
        return nullcontext()
    import memray  # noqa: PLC0415

    return memray.Tracker(
        path,
        native_traces=True,
        file_format=memray.FileFormat.AGGREGATED_ALLOCATIONS,
    )


async def run(args: argparse.Namespace) -> None:
    """Boot the hermetic server and serve until SIGINT/SIGTERM."""
    # imported late so PERF_YAPPI_BOOT profiling can start before this import
    from music_assistant_models.auth import UserRole  # noqa: PLC0415

    from music_assistant.controllers.config.controller import ConfigController  # noqa: PLC0415
    from music_assistant.mass import MusicAssistant  # noqa: PLC0415

    data_dir = str(Path(args.data_dir).resolve())
    cache_dir = os.path.join(data_dir, ".cache")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    if args.fingerprint_fixture:
        await install_fingerprint_fixture(Path(data_dir))

    # force local-only webserver/streamserver binding before anything reads the config
    orig_cfg_setup = ConfigController.setup

    async def patched_cfg_setup(self: ConfigController) -> None:
        await orig_cfg_setup(self)
        # dedicated stream port per instance: the default (8097) is shared with any
        # other MA instance on this machine, so streams could silently end up served
        # by the wrong process
        for core_module, key, value in (
            ("webserver", "bind_ip", "127.0.0.1"),
            ("webserver", "bind_port", args.port),
            ("webserver", "base_url", f"http://127.0.0.1:{args.port}"),
            ("streams", "bind_ip", "127.0.0.1"),
            ("streams", "publish_ip", "127.0.0.1"),
            ("streams", "bind_port", args.stream_port),
        ):
            self.set_raw_core_config_value(core_module, key, value)

    ConfigController.setup = patched_cfg_setup  # type: ignore[method-assign]

    # hermetic: strip every builtin provider manifest not on the allowlist before
    # builtin load. Most importantly sendspin: it runs its own aiosendspin server with
    # its own mDNS advertisement (bypassing the mocked discovery controller) and real
    # devices on the LAN WILL discover and connect to the test instance otherwise.
    orig_load_builtin = MusicAssistant._load_builtin_providers

    async def patched_load_builtin(self: MusicAssistant) -> None:
        for domain, manifest in list(self._provider_manifests.items()):
            if manifest.builtin and domain not in ALLOWED_BUILTIN_PROVIDERS:
                self._provider_manifests.pop(domain)
        await orig_load_builtin(self)

    MusicAssistant._load_builtin_providers = patched_load_builtin  # type: ignore[method-assign]

    patches = [
        patch(
            "music_assistant.controllers.discovery.controller.AsyncZeroconf",
            return_value=_create_mock_zeroconf(),
        ),
        patch(
            "music_assistant.controllers.discovery.controller.AsyncServiceBrowser",
            return_value=NonCallableMagicMock(),
        ),
        # hermetic: no real SSDP search
        patch(
            "music_assistant.controllers.discovery.controller.async_upnp_search",
            new=AsyncMock(),
        ),
        # hermetic: no auto-loaded device providers (dlna/sonos/...)
        patch("music_assistant.mass.DEFAULT_PROVIDERS", set()),
    ]
    for patcher in patches:
        patcher.start()

    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=32))

    mass = MusicAssistant(data_dir, cache_dir)
    # dev mode allows loading the `_`-prefixed demo player provider
    mass.dev_mode = True

    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    t_start = time.monotonic()
    await mass.start()
    startup_secs = time.monotonic() - t_start
    # re-apply production-like log level (MA config may have set VERBOSE in dev mode)
    logging.getLogger("music_assistant").setLevel(getattr(logging, args.log_level.upper()))

    # admin user + long-lived token for the bench client
    token_file = Path(data_dir) / "token.txt"
    if not token_file.is_file():
        user = await mass.webserver.auth.create_user("perfadmin", role=UserRole.ADMIN)
        token = await mass.webserver.auth.create_token(user, "perfbench", is_long_lived=True)
        await asyncio.to_thread(token_file.write_text, token)

    lag_monitor = LoopLagMonitor()
    mass.create_task(lag_monitor.run())
    control_api = ControlApi(mass, lag_monitor)
    with allocation_tracking(args.allocation_profile):
        await control_api.start(args.control_port)

        print(
            f"PERF_SERVER_READY pid={os.getpid()} port={args.port} "
            f"control_port={args.control_port} stream_port={args.stream_port} "
            f"startup_secs={startup_secs:.2f} "
            f"rss_mb={psutil.Process().memory_info().rss / 1024 / 1024:.1f}",
            flush=True,
        )

        await stop_event.wait()
        LOGGER.info("shutting down")
        await control_api.stop()
        await mass.stop()
        for patcher in patches:
            patcher.stop()


def main() -> None:
    """Parse arguments and run the hermetic server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--control-port", type=int, default=8099)
    parser.add_argument("--stream-port", type=int, default=8100)
    parser.add_argument("--log-level", default="warning")
    parser.add_argument("--fingerprint-fixture", action="store_true")
    parser.add_argument("--allocation-profile", type=Path)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    logging.getLogger("zeroconf").setLevel(logging.WARNING)

    if os.environ.get("PERF_YAPPI_BOOT") == "1":
        yappi.set_clock_type("cpu")
        yappi.start(builtins=False)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
