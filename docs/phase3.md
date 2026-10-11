# Phase 3 Implementation Record

## Objective

Add an operational single-site boundary around the local MVP: a versioned API, durable source registration, health and metrics endpoints, optional API-key protection, webhook notifications, and local Docker deployment.

## Delivered

- Added `api_server.py` with FastAPI app factory and `/api/v1` endpoints.
- Added health endpoint at `/health`.
- Added run listing, alert listing and mutation, source registration/status, and metrics endpoints.
- Added optional `RESCUE_EYE_API_KEY` enforcement through `X-API-Key`.
- Added webhook notification delivery for a selected alert.
- Added durable `sources` table and source types: `local_file`, `rtsp`, and `webcam`.
- Added `Dockerfile`, `docker-compose.yml`, and `.dockerignore` for CPU-compatible local deployment.
- Added API and source-registry tests.

## API Commands

Run locally from the repository root:

```powershell
$env:RESCUE_EYE_API_KEY = "local-development-key"
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Check health:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Register a source:

```powershell
$headers = @{ "X-API-Key" = $env:RESCUE_EYE_API_KEY }
$body = @{ name = "Training camera"; source_type = "rtsp"; uri = "rtsp://camera.local/live" } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/v1/sources -Method Post -Headers $headers -ContentType "application/json" -Body $body
```

List alerts:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/alerts -Headers $headers
```

Run the local deployment:

```powershell
docker compose up --build
```

- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`

## Scope Boundary

This phase registers and monitors source metadata but does not yet ingest RTSP frames. The existing pipeline still processes local files. Live RTSP/webcam capture, reconnect, buffering, and store-and-forward belong to the next source-adapter increment.

The API key is suitable for local development only. Production deployment requires a secret manager, TLS, user identity, role-based authorization, rate limits, and audit correlation IDs.

## Validation

- API tests run without a GPU, camera, Arduino, database server, or model download.
- Docker services share the local `data`, `models`, and `Ultralytics` directories.
- The default Compose key is intentionally a development fallback and must be overridden before any networked deployment.

## Executed Validation

On 2026-10-11:

- The complete test suite passed with 6 tests.
- Python compilation passed for the API, pipeline, storage, dashboard, and tests.
- The API health endpoint returned HTTP 200 with an initialized file-backed SQLite database.
- `docker compose config` passed and rendered both API and dashboard services.
- API-key protection returned HTTP 401 when the key was missing during endpoint tests.
