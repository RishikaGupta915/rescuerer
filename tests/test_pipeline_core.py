from __future__ import annotations

import numpy as np

from geo.geo_mapper import GeoMapper
from motion.motion_analyzer import MotionAnalyzer
from pose.pose_analyzer import PoseResult
from tracking.tracker import TrackedPerson
from triage.severity_scorer import SeverityScorer
from storage import RunStore
from fastapi.testclient import TestClient
from api_server import create_app


def make_track(track_id: int = 1) -> TrackedPerson:
    return TrackedPerson(
        track_id=track_id,
        x1=10,
        y1=10,
        x2=70,
        y2=110,
        confidence=0.9,
    )


def make_pose(posture: str, confidence: float = 0.9) -> PoseResult:
    return PoseResult(
        track_id=1,
        posture=posture,
        posture_confidence=confidence,
        landmark_count=0,
        x1=10,
        y1=10,
        x2=70,
        y2=110,
    )


def test_motion_analyzer_initializes_tracks_without_motion() -> None:
    analyzer = MotionAnalyzer()
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    result = analyzer.analyze(frame, [make_track()])

    assert len(result) == 1
    assert result[0].movement_level == "INITIALIZING"
    assert result[0].movement_score == 0.0


def test_severity_scorer_marks_stationary_lying_person_as_candidate() -> None:
    motion_result = MotionAnalyzer().analyze(
        np.zeros((120, 160, 3), dtype=np.uint8),
        [make_track()],
    )[0]

    result = SeverityScorer().score([motion_result], [make_pose("LYING")])[0]

    assert result.victim_candidate is True
    assert result.posture == "LYING"
    assert result.severity in {"MEDIUM", "HIGH"}


def test_geo_mapper_preserves_simulated_coordinates_and_filters_non_candidates() -> None:
    motion_result = MotionAnalyzer().analyze(
        np.zeros((120, 160, 3), dtype=np.uint8),
        [make_track()],
    )[0]
    severity_result = SeverityScorer().score([motion_result], [make_pose("LYING")])[0]

    mapped = GeoMapper().map_results([severity_result], frame_width=160, frame_height=120)

    assert len(mapped) == 1
    assert mapped[0].latitude == 12.9766
    assert mapped[0].longitude == 77.5971


def test_run_store_records_alert_and_audit_lifecycle(tmp_path) -> None:
    motion_result = MotionAnalyzer().analyze(
        np.zeros((120, 160, 3), dtype=np.uint8),
        [make_track()],
    )[0]
    severity_result = SeverityScorer().score([motion_result], [make_pose("LYING")])[0]
    geo_result = GeoMapper().map_results([severity_result], frame_width=160, frame_height=120)[0]
    store = RunStore(tmp_path / "rescue_eye.db")

    run_id = store.record_run(
        {
            "timestamp": "2026-10-11T10:00:00",
            "input_video": "input.mp4",
            "output_video": "output.mp4",
            "geo_output": "snapshot.json",
            "geo_mode": "demo",
        },
        [geo_result],
    )

    alerts = store.list_alerts(run_id)
    assert len(alerts) == 1
    assert alerts[0]["status"] == "NEW"
    store.update_alert(alerts[0]["alert_id"], "ACKNOWLEDGED", actor="tester", note="Reviewed evidence")

    updated = store.list_alerts(run_id)[0]
    assert updated["status"] == "ACKNOWLEDGED"
    assert updated["operator_note"] == "Reviewed evidence"
    assert store.count() == {"runs": 1, "observations": 1, "alerts": 1}


def test_run_store_registers_and_updates_sources(tmp_path) -> None:
    store = RunStore(tmp_path / "rescue_eye.db")

    source = store.register_source("Training camera", "rtsp", "rtsp://camera.local/live")
    assert source["name"] == "Training camera"
    assert source["status"] == "registered"

    updated = store.update_source_status(source["source_id"], "online")
    assert updated["status"] == "online"
    assert store.list_sources()[0]["source_id"] == source["source_id"]


def test_api_exposes_health_sources_and_protected_alerts(tmp_path) -> None:
    client = TestClient(create_app(tmp_path / "api.db", api_key="test-key"))

    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/alerts").status_code == 401

    source_response = client.post(
        "/api/v1/sources",
        headers={"X-API-Key": "test-key"},
        json={"name": "Command camera", "source_type": "local_file", "uri": "data/video.mp4"},
    )
    assert source_response.status_code == 201
    assert source_response.json()["source_type"] == "local_file"
    assert client.get("/api/v1/sources", headers={"X-API-Key": "test-key"}).json()[0]["name"] == "Command camera"
