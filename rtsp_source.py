from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2

SOURCE_TYPE_RTSP = "rtsp"


class RtspSourceError(RuntimeError):
    """Raised when RTSP source operations fail."""


class RtspFrameSource:
    """Manages an RTSP video source frame capture with reconnect logic.

    Handles RTSP stream connection, frame retrieval, status tracking,
    and automatic reconnect on failure. Uses OpenCV VideoCapture under
    the hood, which supports RTSP/RTSP URLs with ffmpeg or gstreamer backends.
    """

    def __init__(
        self,
        uri: str,
        timeout: float = 5.0,
        reconnect_delay: float = 2.0,
        max_reconnect_attempts: int = 10,
    ) -> None:
        self.uri: str = uri
        self.timeout: float = timeout
        self.reconnect_delay: float = reconnect_delay
        self.max_reconnect_attempts: int = max_reconnect_attempts
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_open: bool = False
        self._frame_count: int = 0
        self._consecutive_failures: int = 0
        self._last_status: str = "initialized"
        self._last_frame_time: float = 0.0

    def open(self) -> bool:
        """Open the RTSP stream connection.

        Returns:
            True if the stream was successfully opened, False otherwise.
        """
        try:
            self._cap = cv2.VideoCapture(self.uri)
            # Try common RTSP backends
            backends = [
                cv2.CAP_FFMPEG,
                cv2.CAP_MSMF,
                cv2.CAP_GSTREAMER,
            ]
            for backend in backends:
                self._cap.set(cv2.CAP_PROP_BACKEND_TYPE, backend)
                if self._cap.isOpened():
                    self._is_open = True
                    self._consecutive_failures = 0
                    self._last_status = "opened"
                    print(f"RTSP source opened with backend {backend}: {self.uri}")
                    return True
            # If no specific backend worked, try default
            if self._cap.isOpened():
                self._is_open = True
                self._consecutive_failures = 0
                self._last_status = "opened"
                print(f"RTSP source opened: {self.uri}")
                return True
            self._release()
            return False
        except Exception as exc:
            print(f"RTSP source open error: {exc}")
            self._last_status = "error"
            return False

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False

    def is_open(self) -> bool:
        return self._is_open and self._cap is not None and self._cap.isOpened()

    def read(self) -> Tuple[bool, Any]:
        """Read a frame from the RTSP stream.

        Returns:
            (ret, frame) tuple where ret indicates if frame was read successfully.
        """
        if not self.is_open():
            return False, None

        try:
            start_time = time.time()
            ret, frame = self._cap.read()
            read_time = time.time() - start_time

            if ret and frame is not None:
                self._frame_count += 1
                self._consecutive_failures = 0
                self._last_frame_time = read_time
                return True, frame
            else:
                self._consecutive_failures += 1
                self._last_status = "stream_lost"
                return False, None
        except Exception as exc:
            self._consecutive_failures += 1
            self._last_status = "error"
            print(f"RTSP read error: {exc}")
            return False, None

    def release(self) -> None:
        self._last_status = "closed"
        self._consecutive_failures = 0
        self._release()

    def get_status(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "is_open": self.is_open(),
            "status": self._last_status,
            "frame_count": self._frame_count,
            "consecutive_failures": self._consecutive_failures,
            "last_frame_time": round(self._last_frame_time, 3),
        }

    def should_reconnect(self) -> bool:
        return (
            self._consecutive_failures >= 2
            and self._consecutive_failures < self.max_reconnect_attempts
        )

    def reset_failure_count(self) -> None:
        self._consecutive_failures = 0


class RtspSourceManager:
    """Manages RTSP sources for the Rescue-Eye pipeline.

    Handles multiple RTSP streams, their lifecycle, status tracking,
    and frame delivery to the pipeline. Each source is independently
    managed with its own reconnect logic.
    """

    def __init__(self, timeout: float = 5.0, reconnect_delay: float = 2.0) -> None:
        self._sources: Dict[str, RtspFrameSource] = {}
        self._timeout: float = timeout
        self._reconnect_delay: float = reconnect_delay

    def add_source(self, source_id: str, uri: str) -> RtspFrameSource:
        if source_id in self._sources:
            raise ValueError(f"Source {source_id} already exists")
        source = RtspFrameSource(uri=uri, timeout=self._timeout)
        self._sources[source_id] = source
        return source

    def get_source(self, source_id: str) -> Optional[RtspFrameSource]:
        return self._sources.get(source_id)

    def remove_source(self, source_id: str) -> bool:
        if source_id in self._sources:
            self._sources[source_id].release()
            del self._sources[source_id]
            return True
        return False

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        return {sid: src.get_status() for sid, src in self._sources.items()}

    def update_all(self) -> Dict[str, Dict[str, Any]]:
        """Update all sources: read frames, handle reconnects.

        Returns a dict mapping source_id to (success, frame) results.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for source_id, source in self._sources.items():
            if not source.is_open():
                # Attempt to reconnect
                if source.should_reconnect():
                    print(f"Reconnecting RTSP source {source_id}...")
                    if source.open():
                        results[source_id] = {"status": "reconnected", "frame": None}
                    else:
                        results[source_id] = {"status": "reconnect_failed"}
                else:
                    results[source_id] = {"status": "closed"}
                continue

            ret, frame = source.read()
            if ret and frame is not None:
                results[source_id] = {"status": "ok", "frame": frame}
            elif source.should_reconnect():
                print(f"Frame lost from {source_id}, attempting reconnect...")
                source.release()
                if source.open():
                    results[source_id] = {"status": "reconnected", "frame": None}
                else:
                    results[source_id] = {"status": "reconnect_failed"}
            else:
                results[source_id] = {"status": "stream_lost", "frame": None}
        return results

    def release_all(self) -> None:
        for source in self._sources.values():
            source.release()
        self._sources.clear()