# Phase 1 Implementation Record

## Objective

Make the existing Rescue-Eye proof of concept reproducible on Windows, testable without rescue hardware, and understandable from a clean checkout.

## Baseline Confirmed

- `main.py` provides the executable six-phase local-video pipeline.
- `dashboard/app.py` provides the local Streamlit dashboard.
- YOLO, DeepSORT, OpenCV, MediaPipe, and dashboard dependencies import successfully in the development environment.
- `python main.py --setup-check` passes when a local video and `models/yolov8n.pt` are present.
- Existing geo output is simulated from bounding-box coordinates.
- `test.py` sends signals to an Arduino and is not a CI test.

## Delivered In Phase 1

- `requirements.txt` records runtime and test dependencies.
- `README.md` documents setup, execution, outputs, tests, hardware boundaries, and scope.
- `pytest.ini` defines the deterministic test discovery path.
- `tests/test_pipeline_core.py` covers motion initialization, severity scoring, and geo mapping.
- `.github/workflows/phase1.yml` compiles the application and runs tests on Windows without model or hardware access.
- `docs/prd.md` defines the product scope and executable acceptance gates.

## Verification Commands

```powershell
python main.py --setup-check
python -m compileall -q main.py detection geo hardware motion pose tracking triage dashboard
python -m pytest -q
```

## Known Limitations

- Binary input videos and model weights are local artifacts excluded from Git.
- The current detector test path still needs a mocked YOLO integration test before model-wrapper behavior is fully covered.
- `main.py` uses local filesystem paths; RTSP and phone/IP-camera inputs require the future source-manager adapter.
- Tracking is currently DeepSORT, not ByteTrack.
- The dashboard remains file-backed and single-process.
- CPU correctness is supported; performance targets must be measured on the target machine.

## Exit Criteria For The Next Phase

Phase 1 is complete when a clean Windows environment can install the manifest, run setup check, run deterministic tests, compile the application, and start the dashboard in an empty-data state. The next implementation phase can then introduce a configuration layer and durable run/observation storage without guessing how the current demo works.
