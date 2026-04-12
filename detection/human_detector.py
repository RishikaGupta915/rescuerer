from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_CONFIG_DIR = PROJECT_ROOT / "Ultralytics"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"


@dataclass(slots=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str


class HumanDetector:
    PERSON_CLASS_ID = 0

    def __init__(self, model_path: str | Path | None = None, confidence_threshold: float = 0.35) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.confidence_threshold = confidence_threshold
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
        ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - import failure path
            raise RuntimeError("Failed to import ultralytics. Make sure the virtual environment is set up correctly.") from exc

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at '{self.model_path}'. "
                "Place 'yolov8n.pt' in the models folder before running phase one."
            )

        self._model = YOLO(str(self.model_path))
        return self._model

    def detect(self, frame) -> List[Detection]:
        model = self._load_model()
        results = model(frame, verbose=False)
        person_detections: List[Detection] = []

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id != self.PERSON_CLASS_ID or confidence < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            person_detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                    class_id=class_id,
                    class_name="person",
                )
            )

        return person_detections

