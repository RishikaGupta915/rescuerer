from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from detection.human_detector import Detection


@dataclass(slots=True)
class TrackedPerson:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str = "person"


class PersonTracker:
    def __init__(self, max_age: int = 30, n_init: int = 3) -> None:
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ModuleNotFoundError as exc:
            if exc.name == "pkg_resources":
                raise RuntimeError(
                    "Tracking requires setuptools because deep_sort_realtime imports pkg_resources. "
                    "Run '.\\venv\\Scripts\\python.exe -m pip install setuptools'."
                ) from exc
            raise

        self._tracker = DeepSort(max_age=max_age, n_init=n_init)

    def update(self, detections: Iterable[Detection], frame) -> List[TrackedPerson]:
        frame_h, frame_w = frame.shape[:2]
        deepsort_detections = []
        for detection in detections:
            deepsort_detections.append(
                (
                    [
                        detection.x1,
                        detection.y1,
                        detection.x2 - detection.x1,
                        detection.y2 - detection.y1,
                    ],
                    detection.confidence,
                    detection.class_name,
                )
            )

        tracks = self._tracker.update_tracks(deepsort_detections, frame=frame)
        tracked_people: List[TrackedPerson] = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = map(int, track.to_ltrb())
            left = min(x1, x2)
            top = min(y1, y2)
            right = max(x1, x2)
            bottom = max(y1, y2)

            left = max(0, min(left, frame_w - 1))
            top = max(0, min(top, frame_h - 1))
            right = max(0, min(right, frame_w - 1))
            bottom = max(0, min(bottom, frame_h - 1))

            if right <= left or bottom <= top:
                continue

            tracked_people.append(
                TrackedPerson(
                    track_id=int(track.track_id),
                    x1=left,
                    y1=top,
                    x2=right,
                    y2=bottom,
                    confidence=float(getattr(track, "det_conf", 0.0) or 0.0),
                )
            )

        return tracked_people
