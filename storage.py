from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from geo.geo_mapper import GeoResult

ALERT_STATUSES = ("NEW", "ACKNOWLEDGED", "ASSIGNED", "RESOLVED", "DISMISSED")


class RunStore:
    """Small local SQLite store for Phase 2 runs, observations, and alerts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    input_video TEXT NOT NULL,
                    output_video TEXT NOT NULL,
                    geo_output TEXT NOT NULL,
                    geo_mode TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS observations (
                    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    track_id INTEGER NOT NULL,
                    severity TEXT NOT NULL,
                    severity_score REAL NOT NULL,
                    posture TEXT NOT NULL,
                    movement_level TEXT NOT NULL,
                    movement_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    location_mode TEXT NOT NULL,
                    evidence_ref TEXT NOT NULL,
                    observation_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observation_id INTEGER NOT NULL UNIQUE REFERENCES observations(observation_id) ON DELETE CASCADE,
                    status TEXT NOT NULL DEFAULT 'NEW',
                    assignee TEXT,
                    operator_note TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    CHECK (status IN ('NEW', 'ACKNOWLEDGED', 'ASSIGNED', 'RESOLVED', 'DISMISSED'))
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id INTEGER NOT NULL REFERENCES alerts(alert_id) ON DELETE CASCADE,
                    action TEXT NOT NULL,
                    old_status TEXT,
                    new_status TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def record_run(self, summary: dict[str, Any], geo_results: Iterable[GeoResult]) -> str:
        run_id = uuid.uuid4().hex
        timestamp = str(summary.get("timestamp") or self._now())
        results = list(geo_results)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs(run_id, timestamp, input_video, output_video, geo_output, geo_mode, summary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """ ,
                (
                    run_id,
                    timestamp,
                    str(summary.get("input_video", "")),
                    str(summary.get("output_video", "")),
                    str(summary.get("geo_output", "")),
                    str(summary.get("geo_mode", "demo")),
                    json.dumps(summary),
                ),
            )
            for result in results:
                payload = asdict(result)
                cursor = connection.execute(
                    """
                    INSERT INTO observations(
                        run_id, track_id, severity, severity_score, posture, movement_level,
                        movement_score, confidence, latitude, longitude, location_mode,
                        evidence_ref, observation_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        result.track_id,
                        result.severity,
                        result.severity_score,
                        result.posture,
                        result.movement_level,
                        result.movement_score,
                        result.confidence,
                        result.latitude,
                        result.longitude,
                        getattr(result, "location_mode", "demo"),
                        str(summary.get("output_video", "")),
                        json.dumps(payload),
                    ),
                )
                observation_id = cursor.lastrowid
                now = self._now()
                alert_cursor = connection.execute(
                    """
                    INSERT INTO alerts(observation_id, created_at, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (observation_id, now, now),
                )
                connection.execute(
                    """
                    INSERT INTO audit_events(alert_id, action, old_status, new_status, actor, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (alert_cursor.lastrowid, "created", None, "NEW", "system", now),
                )
        return run_id

    def list_alerts(self, run_id: str | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT alerts.alert_id, alerts.status, alerts.assignee, alerts.operator_note,
                   alerts.created_at, alerts.updated_at, observations.observation_id,
                   observations.run_id, observations.track_id, observations.severity,
                   observations.severity_score, observations.posture, observations.movement_level,
                   observations.movement_score, observations.confidence, observations.latitude,
                   observations.longitude, observations.location_mode, observations.evidence_ref
            FROM alerts
            JOIN observations ON observations.observation_id = alerts.observation_id
        """
        parameters: tuple[Any, ...] = ()
        if run_id is not None:
            query += " WHERE observations.run_id = ?"
            parameters = (run_id,)
        query += " ORDER BY CASE alerts.status WHEN 'NEW' THEN 0 WHEN 'ACKNOWLEDGED' THEN 1 WHEN 'ASSIGNED' THEN 2 ELSE 3 END, observations.severity_score DESC"
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, parameters).fetchall()]

    def update_alert(self, alert_id: int, status: str, actor: str = "operator", note: str | None = None, assignee: str | None = None) -> None:
        if status not in ALERT_STATUSES:
            raise ValueError(f"Unsupported alert status: {status}")
        now = self._now()
        with self._connect() as connection:
            row = connection.execute("SELECT status FROM alerts WHERE alert_id = ?", (alert_id,)).fetchone()
            if row is None:
                raise KeyError(f"Alert {alert_id} does not exist")
            old_status = str(row["status"])
            connection.execute(
                "UPDATE alerts SET status = ?, assignee = COALESCE(?, assignee), operator_note = COALESCE(?, operator_note), updated_at = ? WHERE alert_id = ?",
                (status, assignee, note, now, alert_id),
            )
            connection.execute(
                """
                INSERT INTO audit_events(alert_id, action, old_status, new_status, actor, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (alert_id, "status_changed", old_status, status, actor, now),
            )

    def count(self) -> dict[str, int]:
        with self._connect() as connection:
            run_count = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            observation_count = connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
            alert_count = connection.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        return {"runs": run_count, "observations": observation_count, "alerts": alert_count}
