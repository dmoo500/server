"""Compare isolated real MA streaming with and without full-track fingerprint analysis."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import aiohttp

from scripts.perf.run_benchmark import (
    STREAM_RATE_BYTES_PER_SEC,
    BenchmarkError,
    ControlClient,
    FfmpegCpuSampler,
    PeakRssSampler,
    _consume_stream,
    cpu_seconds,
    start_flow_playback,
    start_server,
    stop_playback,
)
from scripts.perf.ws_client import PerfWsClient, summarize_latencies


async def configure_fixture(
    ws: PerfWsClient, control: ControlClient, enabled: bool
) -> tuple[list[str], set[str]]:
    """Set up the six-track library and wait for its initial synchronization."""
    await control.post("/fingerprints/setup", {"enabled": enabled})
    deadline = time.monotonic() + 60
    tracks = []
    while time.monotonic() < deadline:
        tracks = (await ws.command("music/tracks/library_items", {"limit": 6})).result
        if len(tracks) == 6:
            break
        await asyncio.sleep(0.5)
    else:
        raise BenchmarkError(f"Fixture library did not finish syncing: {tracks}")
    await control.wait_sync_idle()
    players = [
        player["player_id"]
        for player in (await ws.command("players/all")).result
        if player["player_id"].startswith("demo_")
    ]
    if len(players) != 2:
        raise BenchmarkError(f"Expected two demo players, got {players}")
    item_ids = {
        mapping["item_id"]
        for track in tracks
        for mapping in track["provider_mappings"]
        if mapping["provider_domain"] == "test"
    }
    return players, item_ids


def validate_evidence(enabled: bool, evidence: dict[str, Any], expected_ids: set[str]) -> None:
    """Reject skipped, partial, outdated or unexpected fingerprint analyses."""
    if not enabled:
        if evidence["results"]:
            raise BenchmarkError("Baseline unexpectedly produced Dejavu analyses")
        return
    entries = evidence["results"]
    if len(entries) != len(expected_ids) or {entry["item_id"] for entry in entries} != expected_ids:
        raise BenchmarkError(f"Expected analyses for all fixture tracks: {evidence}")
    if not all(
        entry["full"]
        and entry["landmarks"] > 0
        and entry["format_id"] == "ma-dejavu-landmarks-v2"
        and entry["analysis_version"] == 2
        and entry["duration_seconds"] == 60
        for entry in entries
    ):
        raise BenchmarkError(f"Full-track analysis validation failed: {evidence}")


async def measure(  # noqa: PLR0915
    enabled: bool, directory: Path, duration: int, allocation_profile: Path | None = None
) -> dict[str, Any]:
    """Measure two independent streams, including startup of playback and finalization."""
    server = await asyncio.to_thread(
        start_server, directory, fingerprint_fixture=True, allocation_profile=allocation_profile
    )
    ws = PerfWsClient(server.port, server.token())
    try:
        await ws.connect()
        async with aiohttp.ClientSession() as session:
            control = ControlClient(session, server.control_port)
            players, expected_ids = await configure_fixture(ws, control, enabled)
            await control.post("/gc")
            await control.post("/lag/reset")
            rss = PeakRssSampler(server.ps)
            ffmpeg = FfmpegCpuSampler(server.ps)
            rss.start()
            idle_rss = rss.current_mb()
            ffmpeg.start()
            cpu_start = cpu_seconds(server.ps)
            started = time.perf_counter()
            samples = []
            active_observations = 0
            running = True

            async def probe() -> None:
                nonlocal active_observations
                while running:
                    result = await ws.command("players/all")
                    samples.append(result.duration_seconds * 1000)
                    if (await control.get("/lag"))["playback_active"]:
                        active_observations += 1
                    await asyncio.sleep(0.25)

            probe_task = asyncio.create_task(probe())
            try:
                urls = await start_flow_playback(ws, control, players, server, disjoint_tracks=True)
                async with asyncio.TaskGroup() as tasks:
                    consumers = [tasks.create_task(_consume_stream(url, duration)) for url in urls]
                downloaded = [task.result() for task in consumers]
                if any(size < STREAM_RATE_BYTES_PER_SEC * duration * 0.9 for size in downloaded):
                    raise BenchmarkError(f"Stream throughput below 90% of realtime: {downloaded}")
                running = False
                await probe_task
                cpu_used = cpu_seconds(server.ps) - cpu_start
                wall_seconds = time.perf_counter() - started
                ffmpeg_used = ffmpeg.stop()
                peak_rss = rss.stop()
                after_rss = rss.current_mb()
                max_lag = (await control.get("/lag"))["max_lag_ms"]
                evidence = await control.get("/fingerprints")
                validate_evidence(enabled, evidence, expected_ids)
                if not active_observations:
                    raise BenchmarkError("Production playback-active protection was never observed")
                median, p95 = summarize_latencies(samples)
                return {
                    "enabled": enabled,
                    "wall_seconds": round(wall_seconds, 3),
                    "python_cpu_seconds": round(cpu_used, 3),
                    "ffmpeg_cpu_seconds": ffmpeg_used,
                    "rss_before_playback_mb": round(idle_rss, 1),
                    "rss_after_capture_mb": round(after_rss, 1),
                    "peak_rss_mb": peak_rss,
                    "peak_rss_growth_mb": round(peak_rss - idle_rss, 1),
                    "max_loop_lag_ms": max_lag,
                    "downloaded_bytes": downloaded,
                    "api_median_ms": median,
                    "api_p95_ms": p95,
                    "api_samples": len(samples),
                    "active_stream_observations": active_observations,
                    "analyses": evidence["results"],
                }
            finally:
                running = False
                try:
                    await probe_task
                finally:
                    rss.stop()
                    ffmpeg.stop()
                    await stop_playback(ws, players)
    finally:
        await ws.close()
        await asyncio.to_thread(server.stop)


async def run(
    repeats: int, duration: int, allocation_profile_dir: Path | None = None
) -> dict[str, Any]:
    """Alternate baseline and analysis runs in fresh temporary server instances."""
    runs = []
    with tempfile.TemporaryDirectory(prefix="ma-fingerprint-perf-") as temporary:
        root = Path(temporary)
        for repeat in range(repeats):
            for enabled in (False, True) if repeat % 2 == 0 else (True, False):
                directory = root / f"{repeat}-{'on' if enabled else 'off'}"
                directory.mkdir()
                profile = (
                    allocation_profile_dir / f"{directory.name}.bin"
                    if allocation_profile_dir
                    else None
                )
                runs.append(await measure(enabled, directory, duration, profile))
    metrics = [
        "python_cpu_seconds",
        "ffmpeg_cpu_seconds",
        "peak_rss_mb",
        "rss_before_playback_mb",
        "rss_after_capture_mb",
        "peak_rss_growth_mb",
        "max_loop_lag_ms",
        "api_median_ms",
        "api_p95_ms",
    ]
    medians = {
        mode: {
            metric: statistics.median(row[metric] for row in runs if row["enabled"] == enabled)
            for metric in metrics
        }
        for mode, enabled in [("off", False), ("on", True)]
    }
    return {
        "schema_version": 1,
        "platform": platform.platform(),
        "repeats": repeats,
        "stream_seconds": duration,
        "streams": 2,
        "fixture": "deterministic 60-second non-silent stereo WAV; disjoint track IDs",
        "uploads": False,
        "allocation_profiled": allocation_profile_dir is not None,
        "runs": runs,
        "medians": medians,
        "median_delta": {key: medians["on"][key] - medians["off"][key] for key in metrics},
    }


def main() -> None:
    """Run the comparison and save a reusable JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--seconds", type=int, choices=range(75, 121), default=80)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--allocation-profile-dir",
        type=Path,
        help="Separate Memray native-allocation run; timings are not comparable to normal runs",
    )
    args = parser.parse_args()
    if args.allocation_profile_dir:
        import memray  # noqa: F401, PLC0415

        args.allocation_profile_dir.mkdir(parents=True, exist_ok=True)
    report = asyncio.run(run(args.repeats, args.seconds, args.allocation_profile_dir))
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"medians": report["medians"], "delta": report["median_delta"]}, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
