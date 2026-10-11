# Rescue-Eye

Rescue-Eye is a local computer-vision decision-support pipeline for reviewing rescue video. The current implementation processes one local video at a time and produces annotated video, triage observations, simulated map coordinates, and JSON run history.

This repository is in Phase 1: making the existing proof of concept reproducible and testable. It is not an autonomous rescue system, a medical diagnostic tool, or a drone control system.

## Current Pipeline

1. YOLO person detection
2. DeepSORT track association
3. Optical-flow motion analysis
4. MediaPipe pose estimation with a bounding-box fallback
5. Rule-based severity scoring
6. Explicitly simulated geo mapping and JSON output

Drone input is future scope. Phone/IP-camera URLs are future source-manager work; the current CLI accepts local filesystem paths.

## Prerequisites

- Windows 10 or Windows 11
- Python version supported by the installed Ultralytics and MediaPipe packages
- A local YOLO weights file at `models/yolov8n.pt`
- A supported local video in `data/` or an explicit `--input` path
- Optional: NVIDIA GPU and a compatible CUDA PyTorch build
- Optional: `ffmpeg` for browser-compatible output transcoding
- Optional: Arduino on the configured serial port for hardware alerts

Binary videos and model weights are ignored by Git. They must be supplied locally or through the project's approved artifact storage process.

## Setup

From the repository root in PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For an NVIDIA installation, install the CUDA-enabled PyTorch build recommended by the official PyTorch selector before installing the remaining requirements. GPU support is optional; CPU is the correctness baseline.

Verify the environment and required local assets:

```powershell
.\.venv\Scripts\python.exe main.py --setup-check
```

## Run The Pipeline

The default input resolver selects a supported video under `data/` when no input is supplied:

```powershell
.\.venv\Scripts\python.exe main.py --phase phase6
```

Use explicit paths and options when needed:

```powershell
.\.venv\Scripts\python.exe main.py `
  --phase phase6 `
  --input data\your-video.mp4 `
  --output data\phase6_output.mp4 `
  --weights models\yolov8n.pt `
  --conf 0.35 `
  --severity-sensitivity 0.75
```

Available phases are `phase1` through `phase6`. Disable optional analysis with `--disable-pose` or `--disable-motion`. Use `--display` only on a machine with an available OpenCV windowing session.

The normal Phase 6 run writes the following by default:

- Annotated output video: `data/phase1_detection_output.mp4`
- Geo snapshot: `data/phase6_geo_snapshot.json`
- Recent run history: `data/pipeline_history.json`

Coordinates in the current output are simulated projections from image coordinates. They must not be treated as exact victim locations.

## Run The Dashboard

Start the local Streamlit dashboard from the repository root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

The dashboard can review local pipeline outputs and uploaded videos. It does not require a live camera, drone, database, or cloud service in the current phase.

## Tests

Run deterministic unit tests without hardware or a live model:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Compile-check the application modules:

```powershell
.\.venv\Scripts\python.exe -m compileall -q main.py detection geo hardware motion pose tracking triage dashboard
```

## Hardware Alerts

Arduino alerts are optional and disabled when the serial dependency or board is unavailable. Do not run `test.py` as part of CI: it sends real signals to the configured Arduino port. Hardware testing must be an explicit operator action using the documented wiring and safety procedure.

## Documentation

- [Product requirements](docs/prd.md)
- [Architecture and data flow](docs/architecture.md)
- [CPU/GPU and live-input research](docs/cpu.md)
- [Phase 1 implementation record](docs/phase1.md)
- [Planning notes](docs/new.md)

## Scope Boundaries

The system currently supports local recorded video. Multi-camera ingestion, RTSP/phone source adapters, durable incident storage, authenticated users, real geolocation calibration, thermal fusion, field validation, and drone workflows are planned work described in the PRD.
