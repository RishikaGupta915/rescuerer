from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from motion.motion_analyzer import MotionResult
from pose.pose_analyzer import PoseResult


@dataclass(slots=True)
class SeverityResult:
    track_id: int
    severity: str
    severity_score: float
    victim_candidate: bool
    victim_reason: str
    posture: str
    movement_level: str
    movement_score: float
    posture_confidence: float
    detection_confidence: float
    rationale: str
    x1: int
    y1: int
    x2: int
    y2: int


class SeverityScorer:
    def __init__(self, high_threshold: float = 0.75, medium_threshold: float = 0.4) -> None:
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    def score(self, motion_results: Iterable[MotionResult], pose_results: Iterable[PoseResult]) -> List[SeverityResult]:
        pose_by_track = {result.track_id: result for result in pose_results}
        severity_results: List[SeverityResult] = []

        for motion_result in motion_results:
            pose_result = pose_by_track.get(motion_result.track_id)
            if pose_result is None:
                continue

            severity_score = self._compute_score(motion_result, pose_result)
            severity, rationale = self._classify_severity(severity_score, motion_result, pose_result)
            victim_candidate, victim_reason = self._classify_victim_candidate(severity, motion_result, pose_result)

            severity_results.append(
                SeverityResult(
                    track_id=motion_result.track_id,
                    severity=severity,
                    severity_score=severity_score,
                    victim_candidate=victim_candidate,
                    victim_reason=victim_reason,
                    posture=pose_result.posture,
                    movement_level=motion_result.movement_level,
                    movement_score=motion_result.movement_score,
                    posture_confidence=pose_result.posture_confidence,
                    detection_confidence=motion_result.confidence,
                    rationale=rationale,
                    x1=motion_result.x1,
                    y1=motion_result.y1,
                    x2=motion_result.x2,
                    y2=motion_result.y2,
                )
            )

        return severity_results

    def _compute_score(self, motion_result: MotionResult, pose_result: PoseResult) -> float:
        score = 0.0

        posture_weights = {
            "LYING": 0.65,
            "FALLEN": 0.45,
            "STANDING": 0.1,
            "UNKNOWN": 0.2,
        }
        score += posture_weights.get(pose_result.posture, 0.2)

        if motion_result.movement_level == "LOW":
            score += 0.3
        elif motion_result.movement_level == "MEDIUM":
            score += 0.12
        elif motion_result.movement_level in {"INITIALIZING", "DISABLED"}:
            score += 0.08

        if pose_result.posture == "LYING" and motion_result.movement_score < 0.2:
            score += 0.2
        elif pose_result.posture == "FALLEN" and motion_result.movement_score < 0.15:
            score += 0.15

        # Slightly discount low-confidence posture assignments.
        score -= max(0.0, 0.25 - pose_result.posture_confidence) * 0.2
        return max(0.0, min(score, 1.0))

    def _classify_severity(
        self,
        severity_score: float,
        motion_result: MotionResult,
        pose_result: PoseResult,
    ) -> tuple[str, str]:
        if severity_score >= self.high_threshold:
            return "HIGH", f"{pose_result.posture} posture with {motion_result.movement_level.lower()} movement"
        if severity_score >= self.medium_threshold:
            return "MEDIUM", f"{pose_result.posture} posture requires follow-up"
        return "LOW", f"{motion_result.movement_level.title()} movement with lower immediate concern"

    def _classify_victim_candidate(
        self,
        severity: str,
        motion_result: MotionResult,
        pose_result: PoseResult,
    ) -> tuple[bool, str]:
        box_width = max(0, motion_result.x2 - motion_result.x1)
        box_height = max(0, motion_result.y2 - motion_result.y1)

        if motion_result.confidence < 0.2:
            return False, "low detection confidence"
        if box_width < 25 or box_height < 40:
            return False, "track too small for victim confirmation"
        if pose_result.posture in {"LYING", "FALLEN"} and motion_result.movement_level == "HIGH":
            return False, "posture appears transient due to high movement"
        if pose_result.posture in {"LYING", "FALLEN"}:
            return True, f"{pose_result.posture.lower()} posture"
        if motion_result.movement_level == "LOW" and pose_result.posture != "STANDING":
            return True, "low movement with non-standing posture"
        if severity in {"HIGH", "MEDIUM"} and pose_result.posture != "STANDING":
            return True, f"{severity.lower()} triage score"
        return False, "appears mobile or upright"
