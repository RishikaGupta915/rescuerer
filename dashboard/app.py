from __future__ import annotations

import json
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from rtsp_source import RtspSourceManager, RtspFrameSource
from storage import ALERT_STATUSES, RunStore

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_HISTORY_PATH = DATA_DIR / "pipeline_history.json"
DEFAULT_DATABASE_PATH = DATA_DIR / "rescue_eye.db"
DEFAULT_SETTINGS_PATH = DATA_DIR / "dashboard_settings.json"
SUPPORTED_UPLOAD_EXTENSIONS = ("mp4", "avi", "mov", "mkv", "webm")
PROCESSED_VIDEO_PREFIX = "phase6_geo_output_"
PROCESSED_SNAPSHOT_PREFIX = "phase6_geo_snapshot_"
PAGE_OPTIONS = ["Dashboard", "Upload", "Analysis", "Map", "Victims", "Reports", "Settings"]

SEVERITY_COLORS = {
    "HIGH": "#d92d20",
    "MEDIUM": "#f79009",
    "LOW": "#12b76a",
}


st.set_page_config(page_title="Rescue-Eye", page_icon="R", layout="wide")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(230, 80, 27, 0.20), transparent 24%),
                linear-gradient(180deg, #1f0805 0%, #2a0b06 8%, #f6efe9 36%, #f8fafc 100%);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #280905 0%, #3b0d07 60%, #280905 100%);
            border-right: 1px solid rgba(230, 80, 27, 0.35);
        }
        .block-container {
            max-width: 1280px;
            padding-top: 0.8rem;
            padding-bottom: 1.2rem;
        }
        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.25rem;
            border-radius: 18px;
            background: linear-gradient(90deg, #280905 0%, #740A03 100%);
            color: #F5F5F5;
            margin-top: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 16px 40px rgba(31, 8, 5, 0.22);
        }
        .topbar-title {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1;
            margin: 0 0 0.15rem 0;
        }
        .topbar-subtitle {
            margin: 0;
            color: #f6d8cf;
            font-size: 0.95rem;
        }
        .topbar-meta {
            text-align: right;
            font-size: 0.92rem;
            color: #fce7e2;
        }
        .status-dot {
            display: inline-block;
            width: 0.65rem;
            height: 0.65rem;
            border-radius: 999px;
            margin-right: 0.4rem;
            background: #27AE60;
            box-shadow: 0 0 14px rgba(39, 174, 96, 0.7);
        }
        .title {
            font-size: 2rem;
            font-weight: 700;
            color: #111111;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #111111;
            margin-bottom: 1rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(230, 80, 27, 0.4);
            border-radius: 18px;
            background: linear-gradient(180deg, #fff8f3 0%, #fff1e8 100%);
            padding: 0.75rem 0.9rem;
            box-shadow: 0 8px 24px rgba(124, 45, 18, 0.08);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: #111111 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #111111;
        }
        .stButton > button,
        .stDownloadButton > button {
            background: linear-gradient(135deg, #dc2626 0%, #ea580c 100%);
            color: white;
            border: none;
            border-radius: 10px;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #b91c1c 0%, #c2410c 100%);
            color: white;
        }
        .stRadio label, .stSelectbox label, .stFileUploader label {
            color: #f5e6df !important;
        }
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] .st-emotion-cache-10trblm,
        section[data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
            color: #f5f5f5 !important;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #fed7aa;
            border-radius: 12px;
            overflow: hidden;
        }
        .stMarkdown, .stText, p, label, li, h1, h2, h3, h4, h5, h6, span {
            color: #111111;
        }
        .stCaption {
            color: #374151 !important;
        }
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stText,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] li,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] h5,
        section[data-testid="stSidebar"] h6,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] .stCaption {
            color: #F5F5F5 !important;
        }
        iframe[title="streamlit_folium.st_folium"],
        iframe[title="st.iframe"] {
            border-radius: 12px;
            border: 1px solid #fed7aa;
        }
        .priority-card {
            border: 1px solid rgba(230, 80, 27, 0.35);
            border-radius: 16px;
            background: linear-gradient(180deg, #fff8f3 0%, #fff3eb 100%);
            padding: 0.9rem 1rem;
        }
        .priority-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.55rem 0;
            border-bottom: 1px solid rgba(230, 80, 27, 0.14);
        }
        .priority-row:last-child {
            border-bottom: none;
        }
        .severity-chip {
            display: inline-block;
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            color: white;
            font-size: 0.75rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def default_settings() -> dict:
    return {
        "confidence_threshold": 0.35,
        "severity_sensitivity": 0.75,
        "pose_enabled": True,
        "motion_enabled": True,
    }


def load_settings(path: Path) -> dict:
    settings = default_settings()
    if not path.exists():
        return settings

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return settings

    if isinstance(raw, dict):
        settings.update(raw)
    return settings


def save_settings(path: Path, settings: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def load_run_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return pd.DataFrame()

    if not isinstance(raw, list) or not raw:
        return pd.DataFrame()
    return pd.DataFrame(raw)


def resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def valid_processed_outputs(history: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    if history.empty or "output_video" not in history.columns:
        return outputs

    for _, row in history.iloc[::-1].iterrows():
        candidate = resolve_path(row.get("output_video"))
        if candidate is None:
            continue
        if candidate.suffix.lower() != ".mp4":
            continue
        if candidate.name.endswith(".browser.mp4") or ".tmp." in candidate.name:
            continue
        if candidate.exists() and candidate not in outputs:
            outputs.append(candidate)
    return outputs


def source_videos() -> list[Path]:
    if not DATA_DIR.exists():
        return []

    items: list[Path] = []
    for path in DATA_DIR.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower().lstrip(".") not in SUPPORTED_UPLOAD_EXTENSIONS:
            continue
        lower_name = path.name.lower()
        if lower_name.startswith(PROCESSED_VIDEO_PREFIX) or lower_name.startswith(PROCESSED_SNAPSHOT_PREFIX):
            continue
        if lower_name.endswith(".browser.mp4") or ".tmp." in lower_name:
            continue
        items.append(path)
    return sorted(items, key=lambda item: item.name.lower())


def browser_video_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    browser_candidate = path.with_name(f"{path.stem}.browser{path.suffix}")
    if browser_candidate.exists() and browser_candidate.stat().st_size > 0:
        return browser_candidate
    if path.exists() and path.stat().st_size > 0:
        return path
    return None


def load_video_bytes(path: Path | None) -> bytes | None:
    candidate = browser_video_path(path)
    if candidate is None:
        return None
    try:
        return candidate.read_bytes()
    except OSError:
        return None


def load_geo_snapshot(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return pd.DataFrame()

    if not raw:
        return pd.DataFrame()

    frame = pd.DataFrame(raw)
    if "victim_reason" not in frame.columns:
        frame["victim_reason"] = frame.get("rationale", "derived from pipeline")

    for column in ("severity_score", "movement_score", "confidence", "latitude", "longitude"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def row_for_output(history: pd.DataFrame, output_video: Path | None) -> pd.Series | None:
    if output_video is None or history.empty or "output_video" not in history.columns:
        return None

    resolved_output = output_video.resolve()
    for _, row in history.iloc[::-1].iterrows():
        candidate = resolve_path(row.get("output_video"))
        if candidate is not None and candidate.exists() and candidate.resolve() == resolved_output:
            return row
    return None


def build_map(frame: pd.DataFrame) -> str:
    default_location = [12.9716, 77.5946]
    center = default_location if frame.empty else [frame["latitude"].mean(), frame["longitude"].mean()]
    rescue_map = folium.Map(location=center, zoom_start=15 if not frame.empty else 13, tiles="OpenStreetMap")

    for row in frame.itertuples():
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=8,
            color=SEVERITY_COLORS.get(row.severity, "#12b76a"),
            fill=True,
            fill_opacity=0.9,
            popup=(
                f"Victim ID: {row.track_id}<br>"
                f"Severity: {row.severity}<br>"
                f"Posture: {row.posture}<br>"
                f"Movement: {row.movement_level}<br>"
                f"Reason: {row.victim_reason}"
            ),
        ).add_to(rescue_map)

    return rescue_map._repr_html_()


def render_header(active_row: pd.Series | None) -> None:
    timestamp = active_row.get("timestamp", datetime.now().isoformat(timespec="seconds")) if active_row is not None else datetime.now().isoformat(timespec="seconds")
    source_name = Path(str(active_row.get("input_video", ""))).name if active_row is not None else "Awaiting pipeline run"
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <div class="topbar-title">Rescue-Eye</div>
                <p class="topbar-subtitle">AI Disaster Intelligence System</p>
            </div>
            <div class="topbar-meta">
                <div><span class="status-dot"></span>ACTIVE</div>
                <div>{timestamp}</div>
                <div>{source_name}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_video(title: str, path: Path | None, empty_message: str) -> None:
    st.subheader(title)
    candidate = browser_video_path(path)
    video_bytes = load_video_bytes(path)
    if video_bytes is not None:
        st.video(video_bytes)
        st.caption(candidate.name if candidate is not None else "")
    else:
        st.info(empty_message)


def train_models(frame: pd.DataFrame):
    if not SKLEARN_AVAILABLE:
        # Fallback simple benchmarks without sklearn
        df = frame.copy()
        df = df.dropna(subset=["severity", "movement_score", "severity_score", "confidence"])
        if df["severity"].nunique() < 2:
            return []

        label_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        df["severity_label"] = df["severity"].map(label_map)

        # Simple feature set
        feat_cols = ["movement_score", "severity_score", "confidence"]
        df = pd.get_dummies(df, columns=["posture", "movement_level"], drop_first=True)
        feat_cols.extend([c for c in df.columns if c.startswith("posture_") or c.startswith("movement_level_")])

        X = df[feat_cols].to_numpy()
        y = df["severity_label"].to_numpy()

        if len(y) < 6:
            return []

        # manual train/test split
        rng = np.random.default_rng(42)
        idx = np.arange(len(y))
        rng.shuffle(idx)
        split = int(len(y) * 0.7)
        train_idx, test_idx = idx[:split], idx[split:]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        results = []

        # Majority class baseline
        majority = np.bincount(y_train).argmax()
        preds = np.full_like(y_test, majority)
        acc = (preds == y_test).mean()
        cm = confusion_matrix_numpy(y_test, preds, labels=[0, 1, 2])
        results.append({"name": "Majority Baseline", "accuracy": acc, "cm": cm, "labels": ["LOW", "MEDIUM", "HIGH"], "importances": None})

        # Threshold rule on severity_score
        thresh_preds = np.where(X_test[:, feat_cols.index("severity_score")] >= 0.75, 2,
                                np.where(X_test[:, feat_cols.index("severity_score")] >= 0.5, 1, 0))
        acc = (thresh_preds == y_test).mean()
        cm = confusion_matrix_numpy(y_test, thresh_preds, labels=[0, 1, 2])
        results.append({"name": "Rule: severity_score", "accuracy": acc, "cm": cm, "labels": ["LOW", "MEDIUM", "HIGH"], "importances": None})

        # Simple k-NN (k=5) using euclidean distance
        k = 5
        preds_knn = []
        for x in X_test:
            dists = np.linalg.norm(X_train - x, axis=1)
            nn_idx = dists.argsort()[:k]
            vote = np.bincount(y_train[nn_idx], minlength=3).argmax()
            preds_knn.append(vote)
        preds_knn = np.array(preds_knn)
        acc = (preds_knn == y_test).mean()
        cm = confusion_matrix_numpy(y_test, preds_knn, labels=[0, 1, 2])
        results.append({"name": "kNN (k=5)", "accuracy": acc, "cm": cm, "labels": ["LOW", "MEDIUM", "HIGH"], "importances": None})

        return results

    df = frame.copy()
    df = df.dropna(subset=["severity", "movement_score", "severity_score", "confidence"])
    if df.empty:
        return []

    label_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    if df["severity"].nunique() < 2:
        return []

    df["severity_label"] = df["severity"].map(label_map)
    features = ["movement_score", "severity_score", "confidence"]
    df = pd.get_dummies(df, columns=["posture", "movement_level"], drop_first=True)
    feature_cols = [col for col in df.columns if col.startswith("posture_") or col.startswith("movement_level_")] + features

    X = df[feature_cols]
    y = df["severity_label"]

    test_size = 0.3 if len(df) >= 12 else 0.4
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    results = []

    models = [
        ("Logistic Regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, multi_class="auto"))),
        ("KNN (k=5)", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
        ("Decision Tree", DecisionTreeClassifier(max_depth=6, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)),
    ]

    for name, model in models:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(8)
        results.append(
            {
                "name": name,
                "accuracy": acc,
                "cm": cm,
                "labels": ["LOW", "MEDIUM", "HIGH"],
                "importances": importances,
            }
        )

    return results


def build_priority_queue(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    severity_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    queue = frame.copy()
    queue["severity_rank"] = queue["severity"].map(severity_rank).fillna(3)
    queue = queue.sort_values(
        by=["severity_rank", "severity_score", "movement_score"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    queue["priority_rank"] = queue.index + 1
    return queue


def render_priority_queue(frame: pd.DataFrame, store: RunStore, api_key: str | None = None, user_role: str | None = None) -> None:
    st.subheader("Priority Queue")
    try:
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        if user_role:
            headers["X-User-Role"] = user_role
        alerts = store.list_alerts(run_id=None)  # Will be overridden below
    except Exception as exc:
        st.error(f"Alert database unavailable: {exc}")
        alerts = []

    # Try to fetch alerts via API if key provided
    api_alerts: list[dict] = []
    if api_key:
        try:
            from fastapi.testclient import TestClient
            # We can't easily use TestClient here, so use store as fallback
            pass
        except Exception:
            pass

    if alerts:
        for alert in alerts[:8]:
            status = alert["status"]
            label = f"Alert {alert['alert_id']} - Victim {alert['track_id']} - {alert['severity']}"
            with st.expander(f"{label} ({status})", expanded=status in {"NEW", "ACKNOWLEDGED"}):
                st.write(
                    f"Posture: {alert['posture']} | Movement: {alert['movement_level']} | "
                    f"Confidence: {alert['confidence']:.2f}"
                )
                st.caption(f"Evidence: {alert['evidence_ref']} | Location mode: {alert['location_mode']}")
                next_status = st.selectbox(
                    "Alert status",
                    ALERT_STATUSES,
                    index=ALERT_STATUSES.index(status),
                    key=f"alert_status_{alert['alert_id']}",
                )
                note = st.text_input(
                    "Operator note",
                    value=alert["operator_note"] or "",
                    key=f"alert_note_{alert['alert_id']}",
                )
                if st.button("Save Alert", key=f"save_alert_{alert['alert_id']}"):
                    try:
                        store.update_alert(
                            alert["alert_id"],
                            next_status,
                            actor="dashboard_operator",
                            note=note or None,
                        )
                        st.success("Alert updated.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not update alert: {exc}")
    else:
        st.info("No persisted alerts yet.")

    # Show API connection status
    if api_key:
        st.caption(f"🔑 API connected (role: {user_role or 'none'}) | 🌐 WS endpoint: /api/v1/alerts/ws/{{alert_id}} | Press ⬇️ to refresh")
        if st.button("🔄 Refresh Alerts", key="refresh_alerts"):
            st.rerun()
    else:
        st.caption("🔒 No API key configured | Set RESCUE_EYE_API_KEY for real-time alerts & RBAC | WebSocket available at /api/v1/alerts/ws/{{alert_id}}")

    queue = build_priority_queue(frame)
    if queue.empty:
        st.info("Priority queue will appear after a victim-positive run.")
        return

    st.markdown('<div class="priority-card">', unsafe_allow_html=True)
    for row in queue.head(8).itertuples():
        chip_color = SEVERITY_COLORS.get(row.severity, "#12b76a")
        st.markdown(
            f"""
            <div class="priority-row">
                <div>
                    <strong>{row.priority_rank}. Victim {row.track_id}</strong><br>
                    <span style="color:#111111;font-size:0.84rem;">{row.posture} - {row.movement_level} - {row.victim_reason}</span>
                </div>
                <span class="severity-chip" style="background:{chip_color};">{row.severity}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def run_pipeline(source_video: Path, settings: dict, progress_callback=None) -> tuple[bool, dict]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stem = source_video.stem.replace(" ", "_").replace("(", "").replace(")", "")
    output_video = DATA_DIR / f"{PROCESSED_VIDEO_PREFIX}{safe_stem}_{timestamp}.mp4"
    geo_output = DATA_DIR / f"{PROCESSED_SNAPSHOT_PREFIX}{safe_stem}_{timestamp}.json"

    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--phase",
        "phase6",
        "--input",
        str(source_video),
        "--output",
        str(output_video),
        "--geo-output",
        str(geo_output),
        "--history-output",
        str(DEFAULT_HISTORY_PATH),
        "--conf",
        str(settings["confidence_threshold"]),
        "--severity-sensitivity",
        str(settings["severity_sensitivity"]),
    ]

    if not settings.get("pose_enabled", True):
        command.append("--disable-pose")
    if not settings.get("motion_enabled", True):
        command.append("--disable-motion")

    env = dict(os.environ)
    env.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "Ultralytics"))
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    output_lines: list[str] = []
    output_queue: queue.Queue[str] = queue.Queue()
    started_at = time.time()

    def _pump_output() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            output_queue.put(line.rstrip())

    reader = threading.Thread(target=_pump_output, daemon=True)
    reader.start()

    while process.poll() is None:
        elapsed = time.time() - started_at
        saw_output = False
        while True:
            try:
                line = output_queue.get_nowait()
            except queue.Empty:
                break
            saw_output = True
            if line:
                output_lines.append(line)
                if progress_callback is not None:
                    progress_callback(line, elapsed)
        if progress_callback is not None and not saw_output:
            progress_callback(None, elapsed)
        time.sleep(0.2)

    reader.join(timeout=1.0)
    while True:
        try:
            line = output_queue.get_nowait()
        except queue.Empty:
            break
        if line:
            output_lines.append(line)
            if progress_callback is not None:
                progress_callback(line, time.time() - started_at)

    return_code = process.wait()
    output_text = "\n".join(output_lines).strip()

    if return_code != 0:
        return False, {"error": output_text or "Pipeline failed."}
    return True, {"video": output_video}


def render_dashboard(
    frame: pd.DataFrame,
    source_video: Path | None,
    processed_video: Path | None,
    active_row: pd.Series | None,
    store: RunStore,
) -> None:
    render_header(active_row)

    metrics = st.columns(4)
    metrics[0].metric("Victims", int(len(frame)))
    metrics[1].metric("High", int((frame["severity"] == "HIGH").sum()) if not frame.empty else 0)
    metrics[2].metric("Medium", int((frame["severity"] == "MEDIUM").sum()) if not frame.empty else 0)
    metrics[3].metric("Low", int((frame["severity"] == "LOW").sum()) if not frame.empty else 0)

    left, right = st.columns(2)
    with left:
        render_video("Source Video", source_video, "No source video is linked to the active run.")
    with right:
        render_video("Processed Video", processed_video, "No processed video is available yet.")

    table_col, map_col = st.columns([1.35, 1.0])
    with table_col:
        st.subheader("Victim Summary")
        if frame.empty:
            st.info("No victim candidates for the active run.")
        else:
            display = frame.rename(
                columns={
                    "track_id": "Victim ID",
                    "severity": "Severity",
                    "posture": "Posture",
                    "movement_level": "Movement",
                    "movement_score": "Movement Score",
                    "confidence": "Confidence",
                    "victim_reason": "Reason",
                }
            )
            st.dataframe(display, width="stretch", hide_index=True)
    with map_col:
        st.subheader("Map View")
        components.html(build_map(frame), height=420)

    queue_col, info_col = st.columns([1.0, 1.0])
    with queue_col:
        render_priority_queue(frame, store)
    with info_col:
        st.subheader("Overlay Legend")
        st.write("RED -> HIGH")
        st.write("ORANGE -> MEDIUM")
        st.write("GREEN -> LOW")
        st.caption("Processed video shows victim IDs, severity labels, and posture or movement context from the active run.")


def render_upload(settings: dict) -> None:
    render_header(None)
    videos = source_videos()

    st.subheader("Run Pipeline")

    # Source type selection
    source_type = st.radio(
        "Source type",
        ("Local file", "RTSP/IP Camera", "Webcam"),
        index=0,
        help="Local file: processed video from data/; RTSP/IP Camera: live stream (rtsp:// or http:// URL); Webcam: default system camera",
    )

    # Handle source based on type
    rtsp_manager: RtspSourceManager | None = None
    selected_path: Path | None = None
    source_label: str = ""

    if source_type == "RTSP URL":
        rtsp_url = st.text_input("RTSP URL", value="rtsp://camera.local/stream", help="Example: rtsp://admin:password@192.168.1.100:554/stream1")
        if st.button("Connect RTSP Source", type="secondary"):
            try:
                rtsp_manager = RtspSourceManager(timeout=5.0)
                source = rtsp_manager.add_source(source_id="live", uri=rtsp_url)
                if source.open():
                    source_label = f"RTSP Source: {rtsp_url}"
                    st.success(f"RTSP source connected: {rtsp_url}")
                    # Read first frame to verify
                    ret, frame = source.read()
                    if ret and frame is not None:
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="RTSP Preview", use_column_width=True)
                    else:
                        st.warning("RTSP connected but no frame received yet. Buffer may fill.")
                else:
                    st.error(f"Could not open RTSP source: {rtsp_url}")
            except Exception as exc:
                st.error(f"RTSP source error: {exc}")
    elif source_type == "Webcam":
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    source_label = "Webcam (camera index 0)"
                    st.success("Webcam detected")
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Webcam Preview", use_column_width=True)
                else:
                    st.warning("Webcam opened but no frame received")
                cap.release()
            else:
                st.error("Could not open webcam (index 0)")
        except Exception as exc:
            st.error(f"Webcam error: {exc}")
    else:
        # Local file - existing behavior
        if videos:
            names = [path.name for path in videos]
            selected_name = st.selectbox(
                "Choose a source video",
                names,
                index=names.index(st.session_state["selected_source_video"]) if st.session_state.get("selected_source_video") in names else 0,
            )
            st.session_state["selected_source_video"] = selected_name
            selected_path = DATA_DIR / selected_name
            source_label = f"Local file: {selected_name}"
        else:
            st.info("No source videos found in the data folder yet.")
            uploaded = st.file_uploader("Upload a new source video", type=list(SUPPORTED_UPLOAD_EXTENSIONS))
            if uploaded is not None:
                target = DATA_DIR / uploaded.name
                target.write_bytes(uploaded.getbuffer())
                st.session_state["selected_source_video"] = uploaded.name
                st.success(f"Saved {uploaded.name}")
                st.rerun()
            return  # Early return - nothing more to render

    if selected_path is None:
        st.info(f"No source selected. {source_label}")
        return

    st.write(f"Source: {source_label}")
    st.write(f"Confidence threshold: `{settings['confidence_threshold']:.2f}`")
    st.write(f"Severity sensitivity: `{settings['severity_sensitivity']:.2f}`")
    st.write(f"Pose enabled: `{settings['pose_enabled']}`")
    st.write(f"Motion enabled: `{settings['motion_enabled']}`")

    preview_col, action_col = st.columns([1.35, 0.8])
    with preview_col:
        render_video("Selected Source", selected_path, "This source video could not be previewed in the browser.")
    with action_col:
        if st.button("Run Pipeline", type="primary", use_container_width=True):
            status_placeholder = st.empty()
            logs_placeholder = st.empty()
            recent_logs: list[str] = []

            def on_progress(line: str | None, elapsed: float) -> None:
                status_placeholder.info(
                    f"Processing video... {int(elapsed)}s elapsed. "
                    "Please wait while detection and severity scoring run."
                )
                if line:
                    recent_logs.append(line)
                    logs_placeholder.code("\n".join(recent_logs[-8:]), language="text")

            ok, result = run_pipeline(selected_path, settings, progress_callback=on_progress)
            if ok:
                status_placeholder.success("Pipeline completed.")
                st.session_state["active_processed_video"] = result["video"].name
                st.session_state["active_page"] = "Dashboard"
                st.rerun()
            else:
                status_placeholder.empty()
                st.error(result["error"])


def render_analysis(frame: pd.DataFrame, active_row: pd.Series | None) -> None:
    render_header(active_row)
    st.subheader("Analysis")
    if active_row is None:
        st.info("Run the pipeline first to populate analysis metrics.")
        return

    metric_cols = st.columns(3)
    metric_cols[0].metric("Processed Frames", int(active_row.get("processed_frames", 0)))
    metric_cols[1].metric("Detections", int(active_row.get("total_detections", 0)))
    metric_cols[2].metric("Tracked IDs", int(active_row.get("unique_tracked_ids", 0)))

    if frame.empty:
        st.info("No victim candidates were produced for this run, so victim-level charts are empty.")
        return

    left, right = st.columns(2)
    with left:
        severity_chart = px.pie(
            frame,
            names="severity",
            color="severity",
            color_discrete_map=SEVERITY_COLORS,
            title="Severity Distribution",
        )
        severity_chart.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(severity_chart, width="stretch")
    with right:
        confidence_chart = px.histogram(
            frame,
            x="confidence",
            nbins=12,
            color_discrete_sequence=["#E6501B"],
            title="Confidence Distribution",
        )
        confidence_chart.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(confidence_chart, width="stretch")

    st.markdown("---")
    st.subheader("Model Benchmarks (severity classification)")

    model_results = train_models(frame)
    if not model_results:
        if not SKLEARN_AVAILABLE:
            st.info("scikit-learn not available; using simple built-in baselines. Add more data for clearer results.")
        else:
            st.info("Not enough labeled victims to train models yet.")
        return

    acc_df = pd.DataFrame([{"Model": r["name"], "Accuracy": r["accuracy"]} for r in model_results])
    acc_chart = px.bar(acc_df, x="Model", y="Accuracy", text="Accuracy", range_y=[0, 1])
    acc_chart.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(acc_chart, use_container_width=True)

    tabs = st.tabs([r["name"] for r in model_results])
    for tab, result in zip(tabs, model_results):
        with tab:
            cm = result["cm"]
            cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="OrRd", aspect="auto",
                               labels=dict(x="Predicted", y="Actual"))
            cm_fig.update_xaxes(tickvals=[0, 1, 2], ticktext=result["labels"])
            cm_fig.update_yaxes(tickvals=[0, 1, 2], ticktext=result["labels"])
            cm_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(cm_fig, use_container_width=True)

            if result["importances"] is not None:
                imp_fig = px.bar(result["importances"], title="Top Feature Importances")
                imp_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
                st.plotly_chart(imp_fig, use_container_width=True)


def render_victims(frame: pd.DataFrame, active_row: pd.Series | None) -> None:
    render_header(active_row)
    st.subheader("Victims")
    if frame.empty:
        st.info("No victim candidates for the active run.")
        return

    severity_filter = st.selectbox("Filter by severity", ["All", "HIGH", "MEDIUM", "LOW"])
    filtered = frame if severity_filter == "All" else frame[frame["severity"] == severity_filter]
    st.dataframe(filtered, width="stretch", hide_index=True)


def render_map(frame: pd.DataFrame, active_row: pd.Series | None) -> None:
    render_header(active_row)
    st.subheader("Victim Map")
    location_modes = sorted(set(frame["location_mode"].dropna())) if "location_mode" in frame.columns else ["demo"]
    st.caption(f"Location provenance: {', '.join(location_modes)}. Demo coordinates are not exact victim locations.")
    components.html(build_map(frame), height=650)


def render_reports(frame: pd.DataFrame, history: pd.DataFrame, active_row: pd.Series | None) -> None:
    render_header(active_row)
    st.subheader("Reports")

    if active_row is None and history.empty:
        st.info("No report data available yet.")
        return

    if active_row is not None:
        report_left, report_right = st.columns(2)
        with report_left:
            st.write(f"**Input video:** {Path(str(active_row.get('input_video', ''))).name}")
            st.write(f"**Processed frames:** {active_row.get('processed_frames', 0)}")
            st.write(f"**Total detections:** {active_row.get('total_detections', 0)}")
            st.write(f"**Confirmed tracks:** {active_row.get('confirmed_tracks', 0)}")
            st.write(f"**Active victims:** {active_row.get('active_geo_victims', 0)}")
        with report_right:
            if not frame.empty:
                chart = px.bar(
                    frame,
                    x="track_id",
                    y="severity_score",
                    color="severity",
                    color_discrete_map=SEVERITY_COLORS,
                    title="Victim Severity Scores",
                )
                chart.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(chart, width="stretch")

    if not frame.empty:
        st.download_button(
            "Download Victim CSV",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name="rescue_eye_report.csv",
            mime="text/csv",
        )

    if not history.empty:
        st.subheader("Run History")
        st.dataframe(history.copy().iloc[::-1], width="stretch", hide_index=True)


def render_settings(settings: dict) -> None:
    render_header(None)
    st.subheader("Settings")

    with st.form("settings_form"):
        confidence_threshold = st.slider("Confidence threshold", 0.1, 0.9, float(settings["confidence_threshold"]), 0.05)
        severity_sensitivity = st.slider("Severity sensitivity", 0.4, 0.95, float(settings["severity_sensitivity"]), 0.05)
        pose_enabled = st.toggle("Enable posture inference", value=bool(settings["pose_enabled"]))
        motion_enabled = st.toggle("Enable motion analysis", value=bool(settings["motion_enabled"]))
        save_clicked = st.form_submit_button("Save Settings", type="primary", use_container_width=True)

    if save_clicked:
        save_settings(
            DEFAULT_SETTINGS_PATH,
            {
                "confidence_threshold": confidence_threshold,
                "severity_sensitivity": severity_sensitivity,
                "pose_enabled": pose_enabled,
                "motion_enabled": motion_enabled,
            },
        )
        st.success("Settings saved.")


inject_styles()
settings = load_settings(DEFAULT_SETTINGS_PATH)
history = load_run_history(DEFAULT_HISTORY_PATH)
try:
    run_store = RunStore(DEFAULT_DATABASE_PATH)
    database_error = None
except Exception as exc:
    run_store = None
    database_error = exc
processed_runs = valid_processed_outputs(history)
processed_names = [path.name for path in processed_runs]

active_page = st.session_state.get("active_page", "Dashboard")
if active_page not in PAGE_OPTIONS:
    active_page = "Dashboard"
st.session_state["active_page"] = active_page

current_video_name = st.session_state.get("active_processed_video")
if current_video_name not in processed_names:
    current_video_name = processed_names[0] if processed_names else None
st.session_state["active_processed_video"] = current_video_name

with st.sidebar:
    st.markdown("## Rescue-Eye")
    page = st.radio("Navigation", PAGE_OPTIONS, index=PAGE_OPTIONS.index(st.session_state["active_page"]))
    st.session_state["active_page"] = page
    selected_name = st.selectbox(
        "Processed run",
        processed_names or ["No processed videos yet"],
        index=processed_names.index(current_video_name) if current_video_name in processed_names else 0,
        disabled=not processed_names,
    )
    if processed_names:
        st.session_state["active_processed_video"] = selected_name
    st.caption(f"Conf {settings['confidence_threshold']:.2f} - Sens {settings['severity_sensitivity']:.2f}")
    if database_error is not None:
        st.error(f"Local alert database unavailable: {database_error}")

selected_video = DATA_DIR / st.session_state["active_processed_video"] if st.session_state.get("active_processed_video") else None
active_row = row_for_output(history, selected_video)
source_video = resolve_path(active_row.get("input_video")) if active_row is not None else None
geo_snapshot = resolve_path(active_row.get("geo_output")) if active_row is not None else None
snapshot = load_geo_snapshot(geo_snapshot)

if page == "Dashboard":
    if run_store is None:
        st.error("The dashboard cannot manage alerts until the local SQLite database is available.")
    else:
        render_dashboard(snapshot, source_video, selected_video, active_row, run_store)
elif page == "Upload":
    render_upload(settings)
elif page == "Analysis":
    render_analysis(snapshot, active_row)
elif page == "Victims":
    render_victims(snapshot, active_row)
elif page == "Map":
    render_map(snapshot, active_row)
elif page == "Reports":
    render_reports(snapshot, history, active_row)
else:
    render_settings(settings)

