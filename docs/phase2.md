# Phase 2 Implementation Record

## Objective

Deliver the hardened local MVP described in the PRD: durable local run data, explicit demo geolocation provenance, an alert lifecycle, evidence references, and visible dashboard failure states.

## Delivered

- Added `storage.py` with a local SQLite schema for `runs`, `observations`, `alerts`, and `audit_events`.
- Integrated Phase 6 so every completed run writes its summary and latest victim observations to SQLite.
- Added `--database` to select the local database path.
- Added `--geo-mode` with explicit `demo`, `calibrated`, and `gps` values. Only `demo` is executable; other values fail clearly until their calibration/metadata implementations exist.
- Added `location_mode` to geo results and persisted observation data.
- Added evidence references from alerts to the annotated output video.
- Added dashboard alert controls for status changes and operator notes.
- Added visible database error states in the dashboard.
- Kept JSON history and geo snapshot exports for compatibility and report workflows.
- Added a storage lifecycle test covering run creation, alert creation, acknowledgment, operator notes, and counts.

## Alert Lifecycle

```text
NEW -> ACKNOWLEDGED -> ASSIGNED -> RESOLVED
                         |             |
                         +-> DISMISSED+
```

The dashboard can save any supported status transition. Each transition records actor, old status, new status, and timestamp in `audit_events`.

## Run Commands

From the repository root:

```powershell
python main.py --setup-check
python main.py --phase phase6 --geo-mode demo
python -m streamlit run dashboard\app.py
python -m pytest -q
```

For an isolated database during testing:

```powershell
python main.py --phase phase6 --database data\rescue_eye_test.db --geo-mode demo
```

## Acceptance Evidence

- `python main.py --setup-check` passes when the local video and model assets are available.
- `python -m pytest -q` passes deterministic core and SQLite lifecycle tests.
- The pipeline compiles without importing a live camera or Arduino.
- The dashboard can start without a database file; it creates the local store or displays a visible error.
- A successful Phase 6 run produces JSON exports and a SQLite run with persisted alert records.

## Executed Validation

On 2026-10-11, the full enabled pipeline processed the repository's local 309-frame video with these results:

- 504 person detections
- 1,146 confirmed track observations
- 13 unique track IDs
- 215 HIGH and 15 MEDIUM severity observations
- 1 persisted run, 1 latest victim observation, and 1 `NEW` SQLite alert
- Alert evidence reference: the generated annotated output video
- Location provenance: `demo`
- Dashboard startup: passed on local port 8511

The run also demonstrated the intended degraded behavior: missing `COM9` Arduino hardware and missing `ffmpeg` were reported, while video, JSON, and SQLite outputs completed successfully.

## Known Limitations

- Only simulated image-to-coordinate mapping is implemented.
- The database stores the latest victim snapshot per run, not every per-frame observation.
- Alert deduplication across runs and temporal confirmation are future hardening work.
- The dashboard remains a local Streamlit process without authentication.
- The pipeline still uses DeepSORT; tracker replacement is not part of this phase.