from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import cv2
import numpy as np

from tracking.tracker import TrackedPerson


@dataclass(slots=True)
class MotionResult:
    track_id: int
    movement_score: float
    movement_level: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class MotionAnalyzer:
    def __init__(self, smoothing: float = 0.6) -> None:
        self.prev_gray = None
        self.smoothing = smoothing
        self._track_scores: Dict[int, float] = {}

    def analyze(self, frame, tracks: Iterable[TrackedPerson]) -> List[MotionResult]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return [
                MotionResult(
                    track_id=track.track_id,
                    movement_score=0.0,
                    movement_level="INITIALIZING",
                    confidence=track.confidence,
                    x1=track.x1,
                    y1=track.y1,
                    x2=track.x2,
                    y2=track.y2,
                )
                for track in tracks
            ]

        results: List[MotionResult] = []
        active_track_ids = set()

        for track in tracks:
            active_track_ids.add(track.track_id)
            roi_prev, roi_curr = self._extract_rois(self.prev_gray, gray, track)
            raw_score = self._compute_motion_score(roi_prev, roi_curr)
            smoothed_score = self._smooth_score(track.track_id, raw_score)

            results.append(
                MotionResult(
                    track_id=track.track_id,
                    movement_score=smoothed_score,
                    movement_level=self._classify_motion(smoothed_score),
                    confidence=track.confidence,
                    x1=track.x1,
                    y1=track.y1,
                    x2=track.x2,
                    y2=track.y2,
                )
            )

        self.prev_gray = gray
        self._track_scores = {track_id: self._track_scores[track_id] for track_id in active_track_ids if track_id in self._track_scores}
        return results

    def _extract_rois(self, prev_gray, gray, track: TrackedPerson):
        frame_h, frame_w = gray.shape[:2]
        x1 = max(track.x1, 0)
        y1 = max(track.y1, 0)
        x2 = min(track.x2, frame_w)
        y2 = min(track.y2, frame_h)

        if x2 <= x1 or y2 <= y1:
            return None, None

        return prev_gray[y1:y2, x1:x2], gray[y1:y2, x1:x2]

    def _compute_motion_score(self, roi_prev, roi_curr) -> float:
        if roi_prev is None or roi_curr is None:
            return 0.0
        if roi_prev.size == 0 or roi_curr.size == 0:
            return 0.0

        flow = cv2.calcOpticalFlowFarneback(
            roi_prev,
            roi_curr,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return float(np.mean(magnitude))

    def _smooth_score(self, track_id: int, raw_score: float) -> float:
        prev_score = self._track_scores.get(track_id)
        if prev_score is None:
            smoothed = raw_score
        else:
            smoothed = (self.smoothing * prev_score) + ((1.0 - self.smoothing) * raw_score)

        self._track_scores[track_id] = smoothed
        return smoothed

    @staticmethod
    def _classify_motion(score: float) -> str:
        if score < 0.15:
            return "LOW"
        if score < 0.5:
            return "MEDIUM"
        return "HIGH"
