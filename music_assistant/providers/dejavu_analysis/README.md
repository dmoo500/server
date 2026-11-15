# Dejavu analysis: local development

This experimental provider installs `audio-fingerprint` from an exact revision of [dmoo500/audio-fingerprint](https://github.com/dmoo500/audio-fingerprint). There is no PyPI release. The [receiver](https://github.com/dmoo500/audio-fingerprint-server) is an independent experiment, not an official Music Assistant or OHF service. For this worktree, install it into an isolated, git-ignored directory without changing the shared MA virtual environment (the MA environment already supplies NumPy/SciPy/soxr):

```sh
uv pip install --reinstall --no-deps --target .venv-fingerprint ../music-assistant-fingerprint
export PYTHONPATH="$PWD/.venv-fingerprint"
```

Use MA's existing development Python with that `PYTHONPATH`. The installed package metadata lets the normal provider loader satisfy the exact manifest pin without contacting PyPI. Repeat the local installation after changing the shared library.

The manifest references a published Git commit, not a PyPI release. Do not submit this independent prototype upstream without maintainer agreement and verified compatibility.

The independent library is now distributed as `audio-fingerprint` and imported as `audio_fingerprint`. This package rename does not change the stored `ma-dejavu-landmarks-v1` / `ma-dejavu-landmarks-v2` format IDs, codec ID, fingerprints or analysis version, and does not require reanalysis.

The provider analyzes the PCM already decoded for normal playback, including streaming tracks. It does not fetch additional audio, write decoded audio to disk, or expand background scanning beyond the existing filesystem-only scope. Only finite tracks starting at zero are accepted. Version 2 processes the whole decoded track using overlapping windows of at most 30 seconds, with a hop-aligned stride of approximately 20 seconds, for tracks of at most 30 minutes. The rolling PCM buffer is bounded; decoded audio is never stored. Window overlap supplies landmark context around section boundaries. End-of-stream duration may differ from declared duration by at most one second for rounded metadata; larger mismatches fail rather than publishing partial coverage. The actual PCM duration is recorded.

Local results use the existing audio-analysis storage. Sharing is off by default. When enabled, only fingerprints, sampled coverage and available title/artist/album/recording identifiers are sent. No file paths, source URLs, provider IDs or API keys enter the payload. Library metadata is read locally; external tracks without library metadata may lack identification information.

Consent is recorded for the destination used when analysis starts. Enabling sharing later does not backfill old private results. Changing destinations does not redirect pending uploads. Analysis must complete through MA's normal finalization; cancellation never uploads.

When sharing is enabled and the receiver advertises availability support, the provider checks existing local metadata before sampling. A verified full recording skips analysis and sharing; partial or unknown recordings proceed normally. Recovery checks eligible stored results in batches and skips already-full recordings without deleting them or marking them accepted. Remote errors are logged and do not block analysis; older receivers without this optional capability continue to work. Results are cached briefly per destination, credentials and identity. Sharing disabled never sends this metadata.

Availability requests share the existing outbound pacing. If the receiver returns a rate limit or server error, preflight checks pause for the indicated retry period (at least five minutes) while normal analysis continues; this cooldown does not defer uploads.

Extraction, result construction and serialization use MA's existing low-priority analysis workers, concurrency permits and playback-active solo lock. Controller timeouts can cancel overloaded analysis; playback takes priority and no incomplete reference is published. Full coverage still uses CPU, RAM and storage: it cannot be free of performance cost. Per-track results are capped at 1,000,000 landmarks and 36,000,000 serialized bytes. Tracks exceeding those limits fail explicitly rather than silently omitting passages.

The provider's analysis version is now 2: existing v1 analyses are eligible for reanalysis on the next normal complete playback or existing filesystem-only background scan. No streaming audio is fetched just to backfill fingerprints, and existing analyses are not deleted ahead of successful replacement. Old pending v1 uploads remain recoverable. Upgrade the fingerprint receiver before activating this provider: v2 uploads require the new coverage validation and size limits. Existing v1 server references are retained.

## Local server contract

The new experimental [fingerprint server](https://github.com/dmoo500/audio-fingerprint-server) implements this endpoint locally. The older general analysis server is unchanged:

`POST /api/v1/dejavu/submissions`

- `X-API-Key`: upload authentication (personal key with upload permission in gateway mode).
- `X-Application-Key`: additional application authentication when a fingerprint application token is supplied. This is never a local Music Assistant login token.
- `Idempotency-Key`: SHA-256 of decoded canonical JSON, without `fingerprints_encoding`.
- Body: `format_id`, `sample_rate`, `offset_unit`, `windows`, `fingerprints`, `metadata`, `fingerprints_encoding`. Fingerprints use the shared lossless `sha1-80-offset-delta-v1` binary/delta/base64 encoding.
- Offsets are integer 2,048-sample FFT hops at 44,100 Hz relative to the track.
- `format_id` identifies the shared algorithm and its fixed parameters.
- New full-track documents use `ma-dejavu-landmarks-v2`; the receiver also accepts legacy sampled v1 documents. Landmark extraction parameters remain unchanged.
- The server must validate and durably stage a submission before returning HTTP 202: `{"submission_id": "<idempotency key>", "status": "queued"}`.
- Duplicate submissions must return the same ID and `queued` or `processed`.
- The server must not expose unvalidated submissions to recognition queries.

Persisted analysis is the upload recovery source, with destination/payload-scoped acknowledgements and retry timing in MA's cache. One serialized upload path prevents competing local deliveries. Recovery scans once at startup and is scheduled again only when deliveries are pending. Before checking cached upload acceptance, recovery can fill missing metadata from an unambiguous existing MA-library mapping for the stored source and track ID. Library lookups are batched; no provider request, AcoustID lookup or audio reanalysis is performed. The audio-analysis controller persists the repaired header only if the stored header has not changed concurrently. Acoustic data, actual duration, analysis version, timestamps and the original upload destination remain unchanged. Nonempty metadata is not replaced. Unmatched or still-incomplete results are explicitly reported and remain subject to the receiver's metadata gate. A repaired document has a new submission ID; previously accepted incomplete documents are not deleted from the receiver.

For new analyses, the controller resolves existing library metadata or the exact matching queue track; the provider does not traverse player queues. AcoustID Lookup is a separate Chromaprint-based identification provider and can supply library recording IDs, but its fingerprints are not interchangeable with Dejavu landmark hashes. Requests are serialized and spaced at least one second apart. Timeouts, HTTP 429 and 5xx defer all submissions; other statuses block uploads until the provider is reloaded. Numeric or HTTP-date Retry-After is respected up to one day. A cache clear can cause resubmission, so server idempotency is required. Deleting local analysis also deletes its recoverable pending upload.

The ingestion worker, recognition API and microphone test page live in the separate fingerprint server project, not in this MA provider.

For shared analysis, metadata is read from the existing library entry or the matching resolved queue item when the track is not in the library. This does not fetch the provider item again. Queue metadata is accepted only for the same stream URI, never from another playing/preloaded track.

The shared library's 0.2.2 update reduces temporary FFT/filter allocations without changing the v2 fingerprints or coverage. FFTs run in batches and peak filtering uses overlapping halos, retaining float64 precision and the same neighborhoods. The full spectrogram is never retained; batches are discarded after their peaks have been extracted. Existing analysis and receiver data remain compatible; no version-triggered reanalysis or database migration is required for this optimization.

## Compact storage upgrade (library 0.3.0)

Update the receiver and its gateway capability route before the producer. The provider checks authenticated `GET /api/v1/dejavu/capabilities` before its first upload to a destination. Unsupported receivers stop uploads explicitly. Legacy stored results use compact delivery too, but retain the original canonical document's submission ID, cached acceptance and destination-bound consent.

Existing local fingerprint headers are migrated through the audio-analysis controller. A private SQLite backup is created using its already-open connection, including WAL data. Only this provider's headers change; binary analysis arrays, row IDs, timestamps, analysis versions and recorded upload destinations remain unchanged. Malformed or noncanonical legacy rows are logged and retained. Per-row conversions are idempotent; a failed migration keeps the marker unset for retry on restart and blocks uploads without resetting the database or interrupting playback. Allow disk space for the full audio-analysis backup; backups are not deleted automatically. SQLite may reuse freed pages without making the physical file smaller.

The provider-local `dejavu_analysis_header_version` marker is 1. MA analysis version remains 2, and acoustic formats remain v1/v2: packing changes neither the algorithm nor coverage and triggers no reanalysis or additional audio fetching. Receiver database schema is independently versioned at 3; HTTP API major remains 1.

## Application credentials

Like Apple Music, this provider uses MA's shared `app_var()` resolver for its application credential, with an optional secure provider override. The personal **Upload API key** remains separate. **Application token override**, if filled, takes precedence; otherwise the provider resolves `dejavu_application_key` through `app_var()`. If neither supplies a token, the application header is omitted and legacy authentication stays unchanged. Normal delivery and recovered uploads use the same credential resolution.

For this independent pilot there is no official bundled application key. Supply your own token locally via `MASS_APP_VAR_DEJAVU_APPLICATION_KEY`, or for unbundled source builds via the existing `MASS_APP_VARS_FILE` / `~/.musicassistant/app_vars.json` mechanism. Do not overwrite other app variables when adding your entry. The token is sent to the configured recognition server: supply only a credential intended for that destination. Adding a future official bundled key requires maintainer approval and a trusted destination policy; this change does not modify MA's canonical bundle registry. Tokens are never included in fingerprint payloads, cache identifiers or logs.

The new override field is optional and the existing `api_key` field is unchanged, so existing provider data needs no migration. Saving provider settings reloads the provider. There is no submission-status polling: a valid durable `queued` acknowledgement marks delivery, not completed processing. No additional requests are made after acceptance.
