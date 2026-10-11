from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from storage import ALERT_STATUSES, SOURCE_TYPES, RunStore

DEFAULT_DATABASE_PATH = Path(__file__).resolve().parent / "data" / "rescue_eye.db"


class SourceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    source_type: str
    uri: str = Field(min_length=1, max_length=2000)


class SourceStatusUpdate(BaseModel):
    status: str = Field(min_length=1, max_length=40)


class AlertUpdate(BaseModel):
    status: str
    actor: str = Field(default="api_operator", min_length=1, max_length=120)
    note: str | None = Field(default=None, max_length=2000)
    assignee: str | None = Field(default=None, max_length=120)


class WebhookRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2000)
    alert_id: int


def create_app(database_path: str | Path = DEFAULT_DATABASE_PATH, api_key: str | None = None) -> FastAPI:
    store = RunStore(database_path)
    configured_api_key = api_key if api_key is not None else os.getenv("RESCUE_EYE_API_KEY")
    app = FastAPI(title="Rescue-Eye API", version="0.3.0")

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if configured_api_key and x_api_key != configured_api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    @app.get("/health")
    def health() -> dict:
        try:
            counts = store.count()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc
        return {"status": "ok", "database": str(store.path), "counts": counts}

    @app.get("/api/v1/runs", dependencies=[Depends(require_api_key)])
    def list_runs() -> list[dict]:
        return store.list_runs()

    @app.get("/api/v1/alerts", dependencies=[Depends(require_api_key)])
    def list_alerts(run_id: str | None = None) -> list[dict]:
        return store.list_alerts(run_id)

    @app.patch("/api/v1/alerts/{alert_id}", dependencies=[Depends(require_api_key)])
    def update_alert(alert_id: int, payload: AlertUpdate) -> dict:
        if payload.status not in ALERT_STATUSES:
            raise HTTPException(status_code=400, detail=f"status must be one of {ALERT_STATUSES}")
        try:
            store.update_alert(alert_id, payload.status, payload.actor, payload.note, payload.assignee)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return next(alert for alert in store.list_alerts() if alert["alert_id"] == alert_id)

    @app.get("/api/v1/sources", dependencies=[Depends(require_api_key)])
    def list_sources() -> list[dict]:
        return store.list_sources()

    @app.post("/api/v1/sources", status_code=201, dependencies=[Depends(require_api_key)])
    def register_source(payload: SourceCreate) -> dict:
        if payload.source_type not in SOURCE_TYPES:
            raise HTTPException(status_code=400, detail=f"source_type must be one of {SOURCE_TYPES}")
        try:
            return store.register_source(payload.name, payload.source_type, payload.uri)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.patch("/api/v1/sources/{source_id}", dependencies=[Depends(require_api_key)])
    def update_source(source_id: str, payload: SourceStatusUpdate) -> dict:
        try:
            return store.update_source_status(source_id, payload.status)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/metrics", dependencies=[Depends(require_api_key)])
    def metrics() -> dict:
        return {"store": store.count(), "source_count": len(store.list_sources())}

    @app.post("/api/v1/notifications/webhook", dependencies=[Depends(require_api_key)])
    def send_webhook(payload: WebhookRequest) -> dict:
        alert = next((item for item in store.list_alerts() if item["alert_id"] == payload.alert_id), None)
        if alert is None:
            raise HTTPException(status_code=404, detail="Alert not found")
        body = json.dumps({"event": "rescue_eye.alert", "alert": alert}).encode("utf-8")
        request = Request(payload.url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=5) as response:
                return {"delivered": True, "status_code": response.status, "alert_id": payload.alert_id}
        except (OSError, URLError) as exc:
            raise HTTPException(status_code=502, detail=f"Webhook delivery failed: {exc}") from exc

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=False)
