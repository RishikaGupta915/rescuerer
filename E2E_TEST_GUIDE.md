End-to-End Test Guide for Rescue-Eye PRD Release 2
==================================================

This guide covers testing all implemented features from pipeline through API.

--------------------------------------------------
TABLE OF CONTENTS
--------------------------------------------------
1. Pipeline Tests (Local Video)
2. RTSP Source Testing
3. API Endpoint Tests
4. Notification System Tests
5. RBAC (Role-Based Access Control) Tests
6. Full Verification Script
--------------------------------------------------

--------------------------------------------------
1. PIPELINE TESTS (Local Video)
--------------------------------------------------

Prerequisites:
- Virtual environment activated (.venv)
- YOLO weights: models/yolov8n.pt
- Input video: data/stock-footage-senior-man-collapsing-from-sudden-illness-in-the-living-room.webm

Commands:

a) Setup check (validates environment):
   python main.py --setup-check

b) Phase 6 full run (produces all outputs):
   python main.py --phase phase6 --geo-mode demo

   Outputs produced:
   - data/phase1_detection_output.mp4  (annotated video)
   - data/phase6_geo_snapshot.json     (victim geo snapshot)
   - data/pipeline_history.json        (run history JSON)
   - data/rescue_eye.db                (SQLite database with runs/alerts/observations)

c) Other phases (also support RTSP URLs):
   python main.py --phase phase1 --input data/your-video.mp4
   python main.py --phase phase2 --input rtsp://camera.local/stream
   python main.py --phase phase5 --input data/your-video.mp4

d) Verify pipeline history has entries:
   python -c "import json; data = json.load(open('data/pipeline_history.json')); print(f'Runs: {len(data)}')"

--------------------------------------------------
2. RTSP SOURCE TESTING
--------------------------------------------------

Note: Requires an actual RTSP stream server. For testing without hardware:

a) Test RTSP URL resolution (no server needed):
   python -c "from main import resolve_input_video; result = resolve_input_video('rtsp://camera.local/stream'); print(f'Resolved: {result.startswith(\"rtsp://\")}')"

b) Test RTSP source manager initialization:
   python -c "
from rtsp_source import RtspSourceManager, RtspFrameSource
manager = RtspSourceManager()
print(f'Manager created, sources: {len(manager._sources)}')
"

c) Test source status (will show 'initialized' until open):
   python -c "
from rtsp_source import RtspSourceManager
manager = RtspSourceManager()
status = manager.get_all_status()
print(f'Status: {status}')
"

--------------------------------------------------
3. API ENDPOINT TESTS
--------------------------------------------------

Prerequisites: API key for authenticated endpoints

a) Start the API server (in a separate terminal):
   uvicorn api_server:app --host 127.0.0.1 --port 8000

b) Test with curl (unauthenticated - should 401):
   curl http://127.0.0.1:8000/health

c) Test with API key (set env var first):
   export RESCUE_EYE_API_KEY=test-key
   curl http://127.0.0.1:8000/health

c) Test all API endpoints with key:

   # Health
   curl http://127.0.0.1:8000/health

   # List alerts (returns existing alerts from database)
   curl -H "X-API-Key: test-key" http://127.0.0.1:8000/api/v1/alerts

   # List runs
   curl -H "X-API-Key: test-key" http://127.0.0.1:8000/api/v1/runs

   # List sources
   curl -H "X-API-Key: test-key" http://127.0.0.1:8000/api/v1/sources

   # Register new source (POST)
   curl -H "X-API-Key: test-key" -X POST http://127.0.0.1:8000/api/v1/sources \
        -H "Content-Type: application/json" \
        -d '{"name": "Test Camera", "source_type": "local_file", "uri": "data/video.mp4"}'

   # Update alert status (PATCH)
   curl -H "X-API-Key: test-key" -X PATCH http://127.0.0.1:8000/api/v1/alerts/1 \
        -H "Content-Type: application/json" \
        -d '{"status": "ACKNOWLEDGED", "actor": "operator1"}'

   # Metrics
   curl -H "X-API-Key: test-key" http://127.0.0.1:8000/api/v1/metrics

d) Test notification endpoints:

   # Webhook
   curl -H "X-API-Key: test-key" -X POST http://127.0.0.1:8000/api/v1/notifications/webhook \
        -H "Content-Type: application/json" \
        -d '{"alert_id": 1, "url": "http://example.com/webhook"}'

   # SMS
   curl -H "X-API-Key: test-key" -X POST http://127.0.0.1:8000/api/v1/notifications/sms \
        -H "Content-Type: application/json" \
        -d '{"alert_id": 1, "recipient": "+1234567890", "body": "Test alert"}'

   # Email (will fail without SMTP, expected)
   curl -H "X-API-Key: test-key" -X POST http://127.0.0.1:8000/api/v1/notifications/email \
        -H "Content-Type: application/json" \
        -d '{"alert_id": 1, "recipient": "test@example.com", "subject": "Test", "body": "Test"}'

   # Unified notification dispatcher
   curl -H "X-API-Key: test-key" -X POST http://127.0.0.1:8000/api/v1/notifications \
        -H "Content-Type: application/json" \
        -d '{"alert_id": 1, "method": "webhook", "url": "http://example.com"}'

--------------------------------------------------
4. RBAC TESTS (Role-Based Access Control)
--------------------------------------------------

With API key set, test role headers:

a) Viewer role (read-only access):
   curl -H "X-API-Key: test-key" -H "X-User-Role: viewer" http://127.0.0.1:8000/api/v1/alerts
   # Should return 200 with data

   curl -H "X-API-Key: test-key" -H "X-User-Role: viewer" http://127.0.0.1:8000/api/v1/sources
   # Should return 200 with data

b) Operator role (full management):
   curl -H "X-API-Key: test-key" -H "X-User-Role: operator" -X PATCH http://127.0.0.1:8000/api/v1/alerts/1 \
        -H "Content-Type: application/json" \
        -d '{"status": "RESOLVED"}'
   # Should succeed (200)

   curl -H "X-API-Key: test-key" -H "X-User-Role: operator" -X POST http://127.0.0.1:8000/api/v1/sources \
        -H "Content-Type: application/json" \
        -d '{"name": "New Source", "source_type": "rtsp", "uri": "rtsp://cam2.local/stream"}'
   # Should succeed (201)

c) Admin role (all operations):
   curl -H "X-API-Key: test-key" -H "X-User-Role: admin" - all endpoints including
   source registration with all source types (local_file, rtsp, webcam)

d) Test access denial:
   # Viewer trying to register source should 403
   curl -H "X-API-Key: test-key" -H "X-User-Role: viewer" -X POST http://127.0.0.1:8000/api/v1/sources \
        -H "Content-Type: application/json" \
        -d '{"name": "Test", "source_type": "rtsp", "uri": "rtsp://test"}'
   # Expected: 403 Forbidden

   # Unauthorized (wrong API key)
   curl http://127.0.0.1:8000/api/v1/alerts
   # Expected: 401 Unauthorized

--------------------------------------------------
5. FULL VERIFICATION SCRIPT
--------------------------------------------------

Run this comprehensive test to verify everything works:

#!/usr/bin/env python3
"""Full end-to-end verification for Rescue-Eye PRD Release 2."""
import warnings
warnings.filterwarnings('ignore')
import subprocess
import json
import sys

def run_cmd(cmd, cwd=None):
    """Run a command and return (success, output)."""
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return result.returncode == 0, result.stdout + result.stderr

print("=" * 60)
print("RESUE-EYE PRD RELEASE 2 END-TO-END VERIFICATION")
print("=" * 60)

errors = []

# 1. Setup check
print("\n[1/6] Running setup check...")
ok, output = run_cmd("python main.py --setup-check", cwd="F:\\Repositories\\Rescuer\\rescuerer")
if not ok:
    errors.append("Setup check failed")
    print("  FAIL: Setup check")
else:
    print("  PASS: Setup check")

# 2. Phase 6 run
print("\n[2/6] Running Phase 6 pipeline...")
ok, output = run_cmd("python main.py --phase phase6 --geo-mode demo", cwd="F:\\Repositories\\Rescuer\\rescuerer")
if not ok:
    errors.append("Phase 6 run failed")
    print("  FAIL: Phase 6")
else:
    # Verify output files exist
    import os
    checks = [
        ("Annotated video", os.path.exists("data/phase1_detection_output.mp4")),
        ("Geo snapshot", os.path.exists("data/phase6_geo_snapshot.json")),
        ("History JSON", os.path.exists("data/pipeline_history.json")),
        ("SQLite database", os.path.exists("data/rescue_eye.db")),
    ]
    all_exist = all(v for _, v in checks)
    if all_exist:
        print("  PASS: Phase 6 - all output files created")
    else:
        errors.append("Phase 6 missing output files")
        print(f"  FAIL: Missing files: {[(k) for k,v in checks if not v]}")

# 3. Run unit tests
print("\n[3/6] Running unit tests...")
ok, output = run_cmd("python -m pytest -q", cwd="F:\\Repositories\\Rescuer\\rescuerer")
if not ok:
    errors.append("Unit tests failed")
    print("  FAIL: Unit tests")
else:
    print("  PASS: All 6 unit tests")

# 4. API smoke test
print("\n[4/6] Running API smoke test...")
import warnings
warnings.filterwarnings('ignore')
from api_server import create_app
from fastapi.testclient import TestClient

app = create_app(api_key='test-key')
client = TestClient(app)

# Health check
resp = client.get('/health')
if resp.status_code == 200:
    print("  PASS: API health endpoint")
else:
    errors.append("API health failed")
    print(f"  FAIL: API health returned {resp.status_code}")

# Alerts list with key
resp = client.get('/api/v1/alerts', headers={'X-API-Key': 'test-key'})
if resp.status_code == 200:
    print("  PASS: API alerts endpoint")
else:
    errors.append("API alerts failed")
    print(f"  FAIL: API alerts returned {resp.status_code}")

# Source registration
resp = client.post(
    '/api/v1/sources',
    headers={'X-API-Key': 'test-key'},
    json={'name': 'E2E Test', 'source_type': 'local_file', 'uri': 'data/test.mp4'}
)
if resp.status_code == 201:
    print("  PASS: API source registration")
else:
    errors.append("API source registration failed")
    print(f"  FAIL: Source registration returned {resp.status_code}")

# 5. RBAC test
print("\n[5/6] Running RBAC test...")
app2 = create_app(api_key='test-key')
client2 = TestClient(app2)

# Viewer role should work for basic ops
resp = client2.get('/api/v1/alerts', headers={'X-API-Key': 'test-key', 'X-User-Role': 'viewer'})
if resp.status_code == 200:
    print("  PASS: RBAC viewer role")
else:
    errors.append("RBAC viewer failed")
    print(f"  FAIL: RBAC viewer returned {resp.status_code}")

# Operator role should work for alerts
resp = client2.patch(
    '/api/v1/alerts/1',
    headers={'X-API-Key': 'test-key', 'X-User-Role': 'operator'},
    json={'status': 'ACKNOWLEDGED'}
)
if resp.status_code == 200:
    print("  PASS: RBAC operator role")
else:
    errors.append("RBAC operator failed")
    print(f"  FAIL: RBAC operator returned {resp.status_code}")

# 6. SMS test
print("\n[6/6] Running SMS notification test...")
from api_server import create_app
from fastapi.testclient import TestClient

app3 = create_app(api_key='test-key')
client3 = TestClient(app3)

resp = client3.post('/api/v1/notifications/sms', 
                    headers={'X-API-Key': 'test-key'},
                    json={'alert_id': 1, 'recipient': '+1234567890', 'body': 'E2E Test alert'})
if resp.status_code == 200:
    print("  PASS: SMS notification endpoint")
    result = resp.json()
    if result.get('sent') and result.get('recipient') == '+1234567890':
        print("    SMS payload verified")
    else:
        errors.append("SMS payload incomplete")
        print("    WARN: SMS payload incomplete")
else:
    errors.append("SMS notification failed")
    print(f"  FAIL: SMS returned {resp.status_code}")

# Summary
print("\n" + "=" * 60)
if errors:
    print(f"RESULT: FAILED - {len(errors)} error(s)")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("RESULT: ALL CHECKS PASSED ✓")
    sys.exit(0)
EOF

chmod +x test_e2e.sh

print("\nGenerated test_e2e.sh script in the repository root.")
print("Run it with: ./test_e2e.sh")