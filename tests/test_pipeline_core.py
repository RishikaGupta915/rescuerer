from __future__ import annotations

import numpy as np

from geo.geo_mapper import GeoMapper
from motion.motion_analyzer import MotionAnalyzer
from pose.pose_analyzer import PoseResult
from tracking.tracker import TrackedPerson
from triage.severity_scorer import SeverityScorer


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
