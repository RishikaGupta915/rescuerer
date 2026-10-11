# Rescue-Eye: Computer Vision Pipeline Documentation

## Overview

**Rescue-Eye** is an end‑to‑end computer‑vision pipeline for disaster‑response video analysis. It processes a single input video through six sequential phases, producing per‑frame detections, tracks, motion estimates, posture/fall estimates, severity scores, and geo‑referenced victim snapshots.

The core design is **modular and pipeline‑driven**: each phase consumes the output of the previous phase, adds its own analysis, and passes results forward. This separation allows individual phases (e.g., pose estimation) to be toggled on/off without breaking the rest of the pipeline.

---

## Dataset Model (Internal Data Structures)

The pipeline does not train on a custom dataset; it uses a **pretrained YOLO‑v8n** model for person detection and MediaPipe for pose estimation. The "dataset model" refers to the **in‑memory data structures** that represent each detected person and their derived attributes as they travel through the pipeline.

| Class | File | Purpose |
|-------|------|---------|
| `Detection` | `detection/human_detector.py:15` | Raw YOLO output: `x1,y1,x2,y2`, `confidence`, `class_id`, `class_name` (`person`). |
| `TrackedPerson` | `tracking/tracker.py:10` | Stable identity across frames: `track_id`, bounding box, `confidence`, `class_name`. Produced by DeepSORT. |
| `PoseResult` | `pose/pose_analyzer.py:14` | Per‑track posture classification: `posture` (`STANDING`/`LYING`/`FALLEN`/`UNKNOWN`), `posture_confidence`, `landmark_count`, bounding box. |
| `MotionResult` | `motion/motion_analyzer.py` (inferred) | Per‑track motion level (`LOW`/`MEDIUM`/`HIGH`/`INITIALIZING`) and `movement_score`. |
| `SeverityResult` | `triage/severity_scorer.py` (inferred) | Combined severity label (`HIGH`/`MEDIUM`/`LOW`) and `severity_score`, enriched by motion + pose. |
| `GeoResult` | `geo/geo_mapper.py` (inferred) | Spatially mapped severity + posture per track, plus a simulated latitude/longitude snapshot. |

### Data Flow Summary

1. **Phase 1** – Detector yields a `List[Detection]` per frame.
2. **Phase 2** – `PersonTracker` (`DeepSORT`) associates detections into `TrackedPerson`s, giving each a persistent `track_id`.
3. **Phase 3** – `MotionAnalyzer` computes per‑track motion score/level from optical flow or frame differencing.
4. **Phase 4** – `PoseAnalyzer` (MediaPipe) runs on each track's ROI; if landmarks are found, `_classify_from_landmarks` decides `STANDING`/`LYING`/`FALLEN`.
5. **Phase 5** – `SeverityScorer` combines `motion_score` and `pose_result.posture` into a final severity (`HIGH`/ `MEDIUM`/`LOW`).
6. **Phase 6** – `GeoMapper` projects each victim’s bounding box to a simulated geographic point and writes a JSON snapshot.

All structures are **plain Python dataclasses** using `slots=True`, so they can be serialized to JSON for history tracking or further downstream processing. They are not immutable: later persistence work must define which fields are append-only or versioned.

---

## Architecture Diagram

The following Mermaid flowchart visualizes the six‑phase pipeline and the principal CV models involved.

```mermaid
flowchart TD
    %% Input / Output
    VIDEO[Input Video] --> DET[Phase 1: YOLO‑v8 Human Detector]
    DET --> TRK[Phase 2: DeepSORT Tracker]
    TRK --> MOT[Phase 3: Motion Analyzer]
    MOT --> POSE[Phase 4: MediaPipe Pose Estimator]
    POSE --> SCORE[Phase 5: Severity Scorer]
    SCORE --> GEO[Phase 6: Geo Mapper]
    GEO --> OUTPUT[Annotated Output Video + Geo JSON]
    GEO --> STORE[SQLite Run, Observation, Alert Store]

    %% Style notes
    classDef cv fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;
    classDef proc fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    class DET,TRK,MOT,POSE,SCORE,GEO cv;

    %% Model annotations
    YOLO[YOLO‑v8n (person class)]:::cv
    DEEPSORT[DeepSORT]:::cv
    MEDIAPIPE[MediaPipe Pose]:::cv
    MOTION[Motion Analyzer (optical flow) ]:::proc
    SCORENG[Severity Scorer (rule‑based) ]:::proc
    GEOMAP[Geo Mapper (simulated) ]:::proc
```

### Diagram Description

| Block | Computer‑Vision Component | Primary Output |
|-------|---------------------------|----------------|
| **YOLO‑v8n** | Pre‑trained Ultralytics YOLO model (COCO‑trained) | `Detection` boxes for persons (`class_id=0`) |
| **DeepSORT** | Motion‑aware tracking by appearance + motion prediction | `TrackedPerson` with stable `track_id` |
| **Motion Analyzer** | Simple frame‑difference / optical‑flow heuristic | `movement_level` (`LOW`/`MEDIUM`/`HIGH`) + `movement_score` |
| **MediaPipe Pose** | Landmark‑based pose classification (shoulder‑hip alignment, aspect ratio) | `posture` (`STANDING`/`LYING`/`FALLED`) + `landmark_count` |
| **Severity Scorer** | Rule‑based fusion of motion + posture thresholds | `severity` (`HIGH`/`MEDIUM`/`LOW`) + `severity_score` |
| **Geo Mapper** | Projects bounding‑box center to a simulated lat/lon grid | `GeoResult` per victim, written to JSON snapshot |

The pipeline can be **selectively disabled** per-phase via CLI flags (`--disable-pose`, `--disable-motion`), which substitute identity-preserving placeholders (e.g., `UNKNOWN` posture, `DISABLED` motion). Phase 6 records completed runs and the latest geo observations in the local SQLite store at `data/rescue_eye.db`. The store creates a `NEW` alert for each persisted victim observation and retains operator status transitions in an audit table.

---

## How to Use This Documentation

- **Run a specific phase**: `python main.py --phase phase1` through `--phase phase6`.
- **Toggle components**: `--disable-pose` / `--disable-motion`.
- **Inspect intermediate results**: The pipeline writes a history JSON (`--history-output`) and a geo snapshot (`--geo-output`); both contain the internal data‑structure payloads described above.
- **Inspect intermediate results**: The pipeline writes a history JSON (`--history-output`), a geo snapshot (`--geo-output`), and a SQLite run/alert database (`--database`).
- **Swap models**: Replace the YOLO weights (`models/yolov8n.pt`) or provide a custom `--weights` path; the `HumanDetector` class loads any Ultralytics‑compatible `.pt` file.

---

## References & Related Files

| File | Description |
|------|-------------|
| `detection/human_detector.py` | `Detection` dataclass + `HumanDetector` wrapper around YOLOv8 |
| `tracking/tracker.py` | `TrackedPerson` dataclass + `PersonTracker` (DeepSORT) |
| `pose/pose_analyzer.py` | `PoseResult` dataclass + `PoseAnalyzer` (MediaPipe) |
| `triage/severity_scorer.py` | Severity scoring logic (not fully read, but referenced in `main.py`) |
| `geo/geo_mapper.py` | Geo‑mapping logic + snapshot writer |
| `storage.py` | SQLite run, observation, alert, and audit persistence |
| `main.py` | Orchestrates the six phases; contains `draw_*` helpers for visualization |