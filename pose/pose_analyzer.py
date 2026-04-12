from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import cv2
import mediapipe as mp
import numpy as np

from tracking.tracker import TrackedPerson


@dataclass(slots=True)
class PoseResult:
    track_id: int
    posture: str
    posture_confidence: float
    landmark_count: int
    x1: int
    y1: int
    x2: int
    y2: int


class PoseAnalyzer:
    def __init__(self, min_detection_confidence: float = 0.4, min_tracking_confidence: float = 0.4) -> None:
        self._mp_pose = getattr(getattr(mp, "solutions", None), "pose", None)
        self._pose = None

        if self._mp_pose is not None:
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

    def analyze(self, frame, tracks: Iterable[TrackedPerson], motion_lookup: dict[int, float] | None = None) -> List[PoseResult]:
        results: List[PoseResult] = []
        frame_h, frame_w = frame.shape[:2]
        motion_lookup = motion_lookup or {}

        for track in tracks:
            x1 = max(track.x1, 0)
            y1 = max(track.y1, 0)
            x2 = min(track.x2, frame_w)
            y2 = min(track.y2, frame_h)

            if x2 <= x1 or y2 <= y1:
                results.append(self._empty_result(track, posture="UNKNOWN", confidence=0.0))
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                results.append(self._empty_result(track, posture="UNKNOWN", confidence=0.0))
                continue

            if self._pose is None:
                posture, confidence = self._classify_from_bbox(track, motion_lookup.get(track.track_id))
                results.append(self._empty_result(track, posture=posture, confidence=confidence))
                continue

            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pose_output = self._pose.process(rgb_roi)

            if not pose_output.pose_landmarks:
                posture, confidence = self._classify_from_bbox(track, motion_lookup.get(track.track_id))
                results.append(self._empty_result(track, posture=posture, confidence=confidence))
                continue

            landmarks = pose_output.pose_landmarks.landmark
            posture, confidence = self._classify_from_landmarks(landmarks, track)
            visible_landmarks = sum(1 for landmark in landmarks if landmark.visibility > 0.5)

            results.append(
                PoseResult(
                    track_id=track.track_id,
                    posture=posture,
                    posture_confidence=confidence,
                    landmark_count=visible_landmarks,
                    x1=track.x1,
                    y1=track.y1,
                    x2=track.x2,
                    y2=track.y2,
                )
            )

        return results

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()

    def _empty_result(self, track: TrackedPerson, posture: str, confidence: float) -> PoseResult:
        return PoseResult(
            track_id=track.track_id,
            posture=posture,
            posture_confidence=confidence,
            landmark_count=0,
            x1=track.x1,
            y1=track.y1,
            x2=track.x2,
            y2=track.y2,
        )

    def _classify_from_landmarks(self, landmarks, track: TrackedPerson) -> tuple[str, float]:
        left_shoulder = landmarks[self._mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self._mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self._mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self._mp_pose.PoseLandmark.RIGHT_HIP]
        nose = landmarks[self._mp_pose.PoseLandmark.NOSE]

        shoulder_center = np.array(
            [
                (left_shoulder.x + right_shoulder.x) / 2.0,
                (left_shoulder.y + right_shoulder.y) / 2.0,
            ]
        )
        hip_center = np.array(
            [
                (left_hip.x + right_hip.x) / 2.0,
                (left_hip.y + right_hip.y) / 2.0,
            ]
        )
        torso_vector = hip_center - shoulder_center
        torso_length = float(np.linalg.norm(torso_vector))

        bbox_width = max(track.x2 - track.x1, 1)
        bbox_height = max(track.y2 - track.y1, 1)
        aspect_ratio = bbox_width / bbox_height
        vertical_ratio = abs(float(torso_vector[1])) / max(torso_length, 1e-6)
        head_above_hips = nose.y < hip_center[1]
        avg_visibility = float(
            np.mean(
                [
                    left_shoulder.visibility,
                    right_shoulder.visibility,
                    left_hip.visibility,
                    right_hip.visibility,
                    nose.visibility,
                ]
            )
        )

        if aspect_ratio > 1.2 or vertical_ratio < 0.45:
            return "LYING", max(0.35, avg_visibility)

        if head_above_hips and vertical_ratio > 0.72:
            return "STANDING", max(0.45, avg_visibility)

        return "FALLEN", max(0.35, avg_visibility)

    def _classify_from_bbox(self, track: TrackedPerson, motion_score: float | None) -> tuple[str, float]:
        width = max(track.x2 - track.x1, 1)
        height = max(track.y2 - track.y1, 1)
        aspect_ratio = width / height
        motion_score = 0.0 if motion_score is None else motion_score

        if aspect_ratio > 1.2:
            confidence = 0.45 if motion_score < 0.2 else 0.35
            return "LYING", confidence
        if aspect_ratio < 0.65 and motion_score >= 0.15:
            return "STANDING", 0.35
        return "FALLEN", 0.3
