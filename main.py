from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2

from detection.human_detector import DEFAULT_MODEL_PATH, HumanDetector
from geo.geo_mapper import GeoMapper
from motion.motion_analyzer import MotionAnalyzer, MotionResult
from pose.pose_analyzer import PoseAnalyzer, PoseResult
from tracking.tracker import PersonTracker
from triage.severity_scorer import SeverityScorer


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_VIDEO = PROJECT_ROOT / "data" / "disaster_video.mp4"
DEFAULT_OUTPUT_VIDEO = PROJECT_ROOT / "data" / "phase1_detection_output.mp4"
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
GENERATED_VIDEO_MARKERS = ("phase1_", "phase2_", "phase3_", "_output")
DEFAULT_GEO_SNAPSHOT = PROJECT_ROOT / "data" / "phase6_geo_snapshot.json"
DEFAULT_HISTORY_OUTPUT = PROJECT_ROOT / "data" / "pipeline_history.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescue-Eye video pipeline.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_VIDEO, help="Path to the input disaster video.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_VIDEO, help="Path to save the annotated output video.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to YOLO weights file. Defaults to models/yolov8n.pt.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Minimum confidence threshold for person detections.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show frames in a live OpenCV window while processing.",
    )
    parser.add_argument(
        "--phase",
        choices=["phase1", "phase2", "phase3", "phase4", "phase5", "phase6"],
        default="phase6",
        help="Run human detection, tracking, motion analysis, pose estimation, triage scoring, or geo mapping.",
    )
    parser.add_argument(
        "--geo-output",
        type=Path,
        default=DEFAULT_GEO_SNAPSHOT,
        help="Path to save the latest simulated geo snapshot as JSON.",
    )
    parser.add_argument(
        "--history-output",
        type=Path,
        default=DEFAULT_HISTORY_OUTPUT,
        help="Path to append run history as JSON.",
    )
    parser.add_argument(
        "--severity-sensitivity",
        type=float,
        default=0.75,
        help="Threshold for classifying a victim as HIGH severity.",
    )
    parser.add_argument(
        "--disable-pose",
        action="store_true",
        help="Disable posture inference and fall back to UNKNOWN posture.",
    )
    parser.add_argument(
        "--disable-motion",
        action="store_true",
        help="Disable motion analysis and use neutral motion placeholders.",
    )
    parser.add_argument(
        "--setup-check",
        action="store_true",
        help="Only validate first-run setup and print any missing requirements.",
    )
    return parser.parse_args()


def draw_detections(frame, detections) -> None:
    for detection in detections:
        cv2.rectangle(frame, (detection.x1, detection.y1), (detection.x2, detection.y2), (0, 255, 0), 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (detection.x1, max(detection.y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )


def draw_tracks(frame, tracks) -> None:
    for track in tracks:
        cv2.rectangle(frame, (track.x1, track.y1), (track.x2, track.y2), (255, 0, 0), 2)
        label = f"ID {track.track_id}"
        if track.confidence > 0:
            label = f"{label} {track.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (track.x1, max(track.y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )


def draw_motion_results(frame, motion_results) -> None:
    for result in motion_results:
        color = (0, 255, 0)
        if result.movement_level == "MEDIUM":
            color = (0, 165, 255)
        elif result.movement_level in {"HIGH", "INITIALIZING"}:
            color = (0, 0, 255) if result.movement_level == "HIGH" else (255, 255, 0)

        cv2.rectangle(frame, (result.x1, result.y1), (result.x2, result.y2), color, 2)
        label = f"ID {result.track_id} {result.movement_level} {result.movement_score:.2f}"
        cv2.putText(
            frame,
            label,
            (result.x1, max(result.y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )


def draw_pose_results(frame, pose_results) -> None:
    for result in pose_results:
        color = (0, 255, 0)
        if result.posture == "FALLEN":
            color = (0, 165, 255)
        elif result.posture == "LYING":
            color = (0, 0, 255)
        elif result.posture == "UNKNOWN":
            color = (180, 180, 180)

        cv2.rectangle(frame, (result.x1, result.y1), (result.x2, result.y2), color, 2)
        label = f"ID {result.track_id} {result.posture} {result.posture_confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (result.x1, max(result.y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )


def draw_severity_results(frame, severity_results) -> None:
    for result in severity_results:
        color = (0, 255, 0)
        if result.severity == "MEDIUM":
            color = (0, 165, 255)
        elif result.severity == "HIGH":
            color = (0, 0, 255)

        cv2.rectangle(frame, (result.x1, result.y1), (result.x2, result.y2), color, 2)
        label = f"ID {result.track_id} {result.severity} {result.severity_score:.2f}"
        detail = f"{result.posture}/{result.movement_level}"
        cv2.putText(
            frame,
            label,
            (result.x1, max(result.y1 - 24, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
        cv2.putText(
            frame,
            detail,
            (result.x1, max(result.y1 - 4, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def draw_geo_results(frame, geo_results) -> None:
    for result in geo_results:
        # Green for upright/low risk, red for fallen/lying, amber otherwise
        if result.posture in {"LYING", "FALLEN"}:
            color = (0, 0, 255)
        elif result.severity == "LOW":
            color = (0, 200, 0)
        elif result.severity == "MEDIUM":
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        cv2.rectangle(frame, (result.x1, result.y1), (result.x2, result.y2), color, 2)
        line_one = f"Victim {result.track_id}  {result.severity}"
        line_two = f"{result.posture}  {result.movement_level}  {result.severity_score:.2f}"
        cv2.putText(
            frame,
            line_one,
            (result.x1, max(result.y1 - 24, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
        )
        cv2.putText(
            frame,
            line_two,
            (result.x1, max(result.y1 - 4, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def resolve_input_video(requested_path: Path) -> Path:
    if requested_path.exists():
        return requested_path

    if requested_path == DEFAULT_INPUT_VIDEO:
        data_dir = PROJECT_ROOT / "data"
        candidates = [path for path in data_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS]
        preferred_candidates = sorted(
            [path for path in candidates if not any(marker in path.stem.lower() for marker in GENERATED_VIDEO_MARKERS)]
        )
        generated_candidates = sorted(candidates)

        candidates = preferred_candidates or generated_candidates
        if candidates:
            print(f"Using discovered input video: {candidates[0]}")
            return candidates[0]

    return requested_path


def validate_setup(input_path: Path, weights_path: Path) -> list[str]:
    issues: list[str] = []
    resolved_input_path = resolve_input_video(input_path)

    if not resolved_input_path.exists():
        issues.append(
            f"Missing input video: {input_path}\n"
            "Add a sample disaster video under data/, or run with --input <video_path>."
        )

    if not weights_path.exists():
        issues.append(
            f"Missing YOLO weights: {weights_path}\n"
            "Download or copy 'yolov8n.pt' into the models folder, or run with --weights <weights_path>."
        )

    return issues


def print_setup_status(args: argparse.Namespace) -> bool:
    resolved_input_path = resolve_input_video(args.input)
    issues = validate_setup(args.input, args.weights)
    if not issues:
        print("Setup check passed.")
        print(f"Input video: {resolved_input_path}")
        print(f"YOLO weights: {args.weights}")
        return True

    print("Setup check failed.")
    for issue in issues:
        print()
        print(issue)
    return False


def ensure_video_capture(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open the input video at '{video_path}'.")
    return capture


def build_writer(output_path: Path, fps: float, frame_width: int, frame_height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))


def finalize_output_video(temp_output_path: Path, final_output_path: Path, retries: int = 8, delay_seconds: float = 0.5) -> None:
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for _ in range(retries):
        try:
            temp_output_path.replace(final_output_path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(delay_seconds)

    for _ in range(retries):
        try:
            shutil.copyfile(temp_output_path, final_output_path)
            try:
                temp_output_path.unlink(missing_ok=True)
            except PermissionError:
                pass
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(delay_seconds)

    if last_error is not None:
        raise last_error


def transcode_for_browser(video_path: Path) -> None:
    transcoded_path = video_path.with_name(f"{video_path.stem}.browser{video_path.suffix}")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(transcoded_path),
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("ffmpeg not found. Leaving the OpenCV output unchanged.")
        return

    if completed.returncode != 0 or not transcoded_path.exists():
        error_text = completed.stderr.strip() or completed.stdout.strip()
        print(f"Browser transcode skipped: {error_text}")
        transcoded_path.unlink(missing_ok=True)
        return

    for _ in range(8):
        try:
            transcoded_path.replace(video_path)
            return
        except PermissionError:
            time.sleep(0.5)

    for _ in range(8):
        try:
            shutil.copyfile(transcoded_path, video_path)
            transcoded_path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(0.5)

    print(f"Browser-ready video was created but could not replace the original: {transcoded_path}")


def build_disabled_motion_results(tracks) -> list[MotionResult]:
    return [
        MotionResult(
            track_id=track.track_id,
            movement_score=0.0,
            movement_level="DISABLED",
            confidence=track.confidence,
            x1=track.x1,
            y1=track.y1,
            x2=track.x2,
            y2=track.y2,
        )
        for track in tracks
    ]


def build_disabled_pose_results(tracks) -> list[PoseResult]:
    return [
        PoseResult(
            track_id=track.track_id,
            posture="UNKNOWN",
            posture_confidence=0.0,
            landmark_count=0,
            x1=track.x1,
            y1=track.y1,
            x2=track.x2,
            y2=track.y2,
        )
        for track in tracks
    ]


def append_run_history(history_path: Path, entry: dict) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = []
        except json.JSONDecodeError:
            history = []
    else:
        history = []

    history.append(entry)
    history = history[-25:]
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def run_phase_one(args: argparse.Namespace) -> None:
    input_video = resolve_input_video(args.input)
    detector = HumanDetector(model_path=args.weights, confidence_threshold=args.conf)
    capture = ensure_video_capture(input_video)

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = build_writer(args.output, fps, frame_width, frame_height)

    processed_frames = 0
    total_detections = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            detections = detector.detect(frame)
            draw_detections(frame, detections)

            writer.write(frame)
            processed_frames += 1
            total_detections += len(detections)

            if args.display:
                cv2.imshow("Rescue-Eye Phase 1", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Processed {processed_frames} frames.")
    print(f"Detected {total_detections} people across all frames.")
    print(f"Saved annotated video to: {args.output}")


def run_phase_two(args: argparse.Namespace) -> None:
    input_video = resolve_input_video(args.input)
    detector = HumanDetector(model_path=args.weights, confidence_threshold=args.conf)
    tracker = PersonTracker()
    capture = ensure_video_capture(input_video)

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = build_writer(args.output, fps, frame_width, frame_height)

    processed_frames = 0
    total_detections = 0
    confirmed_tracks = 0
    seen_track_ids: set[int] = set()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)
            draw_tracks(frame, tracks)

            writer.write(frame)
            processed_frames += 1
            total_detections += len(detections)
            confirmed_tracks += len(tracks)
            seen_track_ids.update(track.track_id for track in tracks)

            if args.display:
                cv2.imshow("Rescue-Eye Phase 2", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Processed {processed_frames} frames.")
    print(f"Detected {total_detections} people across all frames.")
    print(f"Generated {confirmed_tracks} confirmed track observations.")
    print(f"Unique tracked IDs: {len(seen_track_ids)}")
    print(f"Saved annotated video to: {args.output}")


def run_phase_three(args: argparse.Namespace) -> None:
    input_video = resolve_input_video(args.input)
    detector = HumanDetector(model_path=args.weights, confidence_threshold=args.conf)
    tracker = PersonTracker()
    motion_analyzer = MotionAnalyzer()
    capture = ensure_video_capture(input_video)

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = build_writer(args.output, fps, frame_width, frame_height)

    processed_frames = 0
    total_detections = 0
    confirmed_tracks = 0
    seen_track_ids: set[int] = set()
    low_motion_observations = 0
    motion_scores: list[float] = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)
            motion_results = motion_analyzer.analyze(frame, tracks)
            draw_motion_results(frame, motion_results)

            writer.write(frame)
            processed_frames += 1
            total_detections += len(detections)
            confirmed_tracks += len(tracks)
            seen_track_ids.update(track.track_id for track in tracks)

            for result in motion_results:
                if result.movement_level == "LOW":
                    low_motion_observations += 1
                if result.movement_level != "INITIALIZING":
                    motion_scores.append(result.movement_score)

            if args.display:
                cv2.imshow("Rescue-Eye Phase 3", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    average_motion = sum(motion_scores) / len(motion_scores) if motion_scores else 0.0
    print(f"Processed {processed_frames} frames.")
    print(f"Detected {total_detections} people across all frames.")
    print(f"Generated {confirmed_tracks} confirmed track observations.")
    print(f"Unique tracked IDs: {len(seen_track_ids)}")
    print(f"Average motion score: {average_motion:.3f}")
    print(f"Low-motion observations: {low_motion_observations}")
    print(f"Saved annotated video to: {args.output}")


def run_phase_four(args: argparse.Namespace) -> None:
    input_video = resolve_input_video(args.input)
    detector = HumanDetector(model_path=args.weights, confidence_threshold=args.conf)
    tracker = PersonTracker()
    motion_analyzer = MotionAnalyzer()
    pose_analyzer = PoseAnalyzer()
    capture = ensure_video_capture(input_video)

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = build_writer(args.output, fps, frame_width, frame_height)

    processed_frames = 0
    total_detections = 0
    confirmed_tracks = 0
    seen_track_ids: set[int] = set()
    posture_counts: dict[str, int] = {"STANDING": 0, "FALLEN": 0, "LYING": 0, "UNKNOWN": 0}
    motion_scores: list[float] = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)
            motion_results = motion_analyzer.analyze(frame, tracks)
            motion_lookup = {result.track_id: result.movement_score for result in motion_results}
            pose_results = pose_analyzer.analyze(frame, tracks, motion_lookup=motion_lookup)
            draw_pose_results(frame, pose_results)

            writer.write(frame)
            processed_frames += 1
            total_detections += len(detections)
            confirmed_tracks += len(tracks)
            seen_track_ids.update(track.track_id for track in tracks)

            for motion_result in motion_results:
                if motion_result.movement_level != "INITIALIZING":
                    motion_scores.append(motion_result.movement_score)

            for pose_result in pose_results:
                posture_counts[pose_result.posture] = posture_counts.get(pose_result.posture, 0) + 1

            if args.display:
                cv2.imshow("Rescue-Eye Phase 4", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        writer.release()
        pose_analyzer.close()
        cv2.destroyAllWindows()

    average_motion = sum(motion_scores) / len(motion_scores) if motion_scores else 0.0
    print(f"Processed {processed_frames} frames.")
    print(f"Detected {total_detections} people across all frames.")
    print(f"Generated {confirmed_tracks} confirmed track observations.")
    print(f"Unique tracked IDs: {len(seen_track_ids)}")
    print(f"Average motion score: {average_motion:.3f}")
    print(f"Standing observations: {posture_counts.get('STANDING', 0)}")
    print(f"Fallen observations: {posture_counts.get('FALLEN', 0)}")
    print(f"Lying observations: {posture_counts.get('LYING', 0)}")
    print(f"Unknown posture observations: {posture_counts.get('UNKNOWN', 0)}")
    print(f"Saved annotated video to: {args.output}")


def run_phase_five(args: argparse.Namespace) -> None:
    input_video = resolve_input_video(args.input)
    detector = HumanDetector(model_path=args.weights, confidence_threshold=args.conf)
    tracker = PersonTracker()
    motion_analyzer = MotionAnalyzer()
    pose_analyzer = PoseAnalyzer()
    severity_scorer = SeverityScorer()
    capture = ensure_video_capture(input_video)

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = build_writer(args.output, fps, frame_width, frame_height)

    processed_frames = 0
    total_detections = 0
    confirmed_tracks = 0
    seen_track_ids: set[int] = set()
    motion_scores: list[float] = []
    severity_counts: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)
            motion_results = motion_analyzer.analyze(frame, tracks)
            motion_lookup = {result.track_id: result.movement_score for result in motion_results}
            pose_results = pose_analyzer.analyze(frame, tracks, motion_lookup=motion_lookup)
            severity_results = severity_scorer.score(motion_results, pose_results)
            draw_severity_results(frame, severity_results)

            writer.write(frame)
            processed_frames += 1
            total_detections += len(detections)
            confirmed_tracks += len(tracks)
            seen_track_ids.update(track.track_id for track in tracks)

            for motion_result in motion_results:
                if motion_result.movement_level != "INITIALIZING":
                    motion_scores.append(motion_result.movement_score)

            for severity_result in severity_results:
                severity_counts[severity_result.severity] = severity_counts.get(severity_result.severity, 0) + 1

            if args.display:
                cv2.imshow("Rescue-Eye Phase 5", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        writer.release()
        pose_analyzer.close()
        cv2.destroyAllWindows()

    average_motion = sum(motion_scores) / len(motion_scores) if motion_scores else 0.0
    print(f"Processed {processed_frames} frames.")
    print(f"Detected {total_detections} people across all frames.")
    print(f"Generated {confirmed_tracks} confirmed track observations.")
    print(f"Unique tracked IDs: {len(seen_track_ids)}")
    print(f"Average motion score: {average_motion:.3f}")
    print(f"High severity observations: {severity_counts.get('HIGH', 0)}")
    print(f"Medium severity observations: {severity_counts.get('MEDIUM', 0)}")
    print(f"Low severity observations: {severity_counts.get('LOW', 0)}")
    print(f"Saved annotated video to: {args.output}")


def run_phase_six(args: argparse.Namespace) -> None:
    input_video = resolve_input_video(args.input)
    detector = HumanDetector(model_path=args.weights, confidence_threshold=args.conf)
    tracker = PersonTracker()
    motion_analyzer = MotionAnalyzer() if not args.disable_motion else None
    pose_analyzer = PoseAnalyzer() if not args.disable_pose else None
    medium_threshold = max(0.2, min(args.severity_sensitivity - 0.2, 0.7))
    severity_scorer = SeverityScorer(
        high_threshold=args.severity_sensitivity,
        medium_threshold=medium_threshold,
    )
    geo_mapper = GeoMapper()
    capture = ensure_video_capture(input_video)

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_output_path = args.output.with_name(f"{args.output.stem}.tmp{args.output.suffix}")
    writer = build_writer(temp_output_path, fps, frame_width, frame_height)

    processed_frames = 0
    total_detections = 0
    confirmed_tracks = 0
    seen_track_ids: set[int] = set()
    severity_counts: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    motion_scores: list[float] = []
    latest_geo_results = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)
            motion_results = (
                motion_analyzer.analyze(frame, tracks)
                if motion_analyzer is not None
                else build_disabled_motion_results(tracks)
            )
            motion_lookup = {result.track_id: result.movement_score for result in motion_results}
            pose_results = (
                pose_analyzer.analyze(frame, tracks, motion_lookup=motion_lookup)
                if pose_analyzer is not None
                else build_disabled_pose_results(tracks)
            )
            severity_results = severity_scorer.score(motion_results, pose_results)
            geo_results = geo_mapper.map_results(severity_results, frame_width=frame_width, frame_height=frame_height)
            if geo_results:
                latest_geo_results = geo_results
            draw_geo_results(frame, geo_results)

            writer.write(frame)
            processed_frames += 1
            total_detections += len(detections)
            confirmed_tracks += len(tracks)
            seen_track_ids.update(track.track_id for track in tracks)

            for motion_result in motion_results:
                if motion_result.movement_level not in {"INITIALIZING", "DISABLED"}:
                    motion_scores.append(motion_result.movement_score)

            for geo_result in geo_results:
                severity_counts[geo_result.severity] = severity_counts.get(geo_result.severity, 0) + 1

            if args.display:
                cv2.imshow("Rescue-Eye Phase 6", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        writer.release()
        if pose_analyzer is not None:
            pose_analyzer.close()
        cv2.destroyAllWindows()

    finalize_output_video(temp_output_path, args.output)
    transcode_for_browser(args.output)
    geo_mapper.write_snapshot(args.geo_output, latest_geo_results)
    average_motion = sum(motion_scores) / len(motion_scores) if motion_scores else 0.0
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_video": str(input_video),
        "output_video": str(args.output),
        "geo_output": str(args.geo_output),
        "confidence_threshold": args.conf,
        "severity_sensitivity": args.severity_sensitivity,
        "pose_enabled": not args.disable_pose,
        "motion_enabled": not args.disable_motion,
        "processed_frames": processed_frames,
        "total_detections": total_detections,
        "confirmed_tracks": confirmed_tracks,
        "unique_tracked_ids": len(seen_track_ids),
        "average_motion_score": round(average_motion, 3),
        "high_severity_observations": severity_counts.get("HIGH", 0),
        "medium_severity_observations": severity_counts.get("MEDIUM", 0),
        "low_severity_observations": severity_counts.get("LOW", 0),
        "active_geo_victims": len(latest_geo_results),
    }
    append_run_history(args.history_output, summary)

    print(f"Processed {processed_frames} frames.")
    print(f"Detected {total_detections} people across all frames.")
    print(f"Generated {confirmed_tracks} confirmed track observations.")
    print(f"Unique tracked IDs: {len(seen_track_ids)}")
    print(f"Average motion score: {average_motion:.3f}")
    print(f"High severity observations: {severity_counts.get('HIGH', 0)}")
    print(f"Medium severity observations: {severity_counts.get('MEDIUM', 0)}")
    print(f"Low severity observations: {severity_counts.get('LOW', 0)}")
    print(f"Saved annotated video to: {args.output}")
    print(f"Saved geo snapshot to: {args.geo_output}")
    print(f"Saved run history to: {args.history_output}")


def main() -> None:
    args = parse_args()
    if not print_setup_status(args):
        raise SystemExit(1)

    if args.setup_check:
        return

    if args.phase == "phase1":
        run_phase_one(args)
        return

    if args.phase == "phase2":
        run_phase_two(args)
        return

    if args.phase == "phase3":
        run_phase_three(args)
        return

    if args.phase == "phase4":
        run_phase_four(args)
        return

    if args.phase == "phase5":
        run_phase_five(args)
        return

    run_phase_six(args)


if __name__ == "__main__":
    main()
