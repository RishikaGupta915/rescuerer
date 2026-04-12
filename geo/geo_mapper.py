from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Iterable, List

from triage.severity_scorer import SeverityResult


@dataclass(slots=True)
class GeoResult:
    track_id: int
    severity: str
    severity_score: float
    victim_candidate: bool
    victim_reason: str
    posture: str
    movement_level: str
    movement_score: float
    latitude: float
    longitude: float
    confidence: float
    rationale: str
    x1: int
    y1: int
    x2: int
    y2: int


class GeoMapper:
    def __init__(
        self,
        base_latitude: float = 12.9716,
        base_longitude: float = 77.5946,
        latitude_span: float = 0.01,
        longitude_span: float = 0.01,
    ) -> None:
        self.base_latitude = base_latitude
        self.base_longitude = base_longitude
        self.latitude_span = latitude_span
        self.longitude_span = longitude_span

    def map_results(
        self,
        severity_results: Iterable[SeverityResult],
        frame_width: int,
        frame_height: int,
    ) -> List[GeoResult]:
        geo_results: List[GeoResult] = []

        for result in severity_results:
            if not result.victim_candidate:
                continue

            center_x = (result.x1 + result.x2) / 2.0
            center_y = (result.y1 + result.y2) / 2.0

            latitude = self.base_latitude + (center_y / max(frame_height, 1)) * self.latitude_span
            longitude = self.base_longitude + (center_x / max(frame_width, 1)) * self.longitude_span

            geo_results.append(
                GeoResult(
                    track_id=result.track_id,
                    severity=result.severity,
                    severity_score=result.severity_score,
                    victim_candidate=result.victim_candidate,
                    victim_reason=result.victim_reason,
                    posture=result.posture,
                    movement_level=result.movement_level,
                    movement_score=result.movement_score,
                    latitude=round(latitude, 6),
                    longitude=round(longitude, 6),
                    confidence=result.detection_confidence,
                    rationale=result.rationale,
                    x1=result.x1,
                    y1=result.y1,
                    x2=result.x2,
                    y2=result.y2,
                )
            )

        return geo_results

    @staticmethod
    def write_snapshot(output_path: Path, geo_results: Iterable[GeoResult]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = [asdict(result) for result in geo_results]
        output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
