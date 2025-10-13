# Backend (Flask + ML + TTN bridge)

This repository combines:
- A Flask app (`main.py`) that serves a root HTML page and registers API blueprints.
- `endpoints/routes.py` — sample API endpoints (GET /api/data).
- `endpoints/ml_backend.py` — ML inference and training endpoints (blueprint `ml_bp`).
- `services/mqtt_bridge.py` — MQTT bridge that writes TTN uplink messages to `public/.../message_history.json` and forwards to ThingSpeak.

Quick local run

1. Activate your virtualenv (Windows PowerShell):
```powershell
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
python -m pip install -r requirements.txt
```

3. Run in development mode:
```powershell
python main.py
# or
python -m waitress --port=8000 main:app
```

Production / deploy

- Render (recommended for this repo):
	1. Push this repository to GitHub.
	2. In Render dashboard, create two services:
		 - Web service: start command `gunicorn main:app -b 0.0.0.0:$PORT --workers 2` (or use the Dockerfile).
		 - Worker service: start command `python services/mqtt_bridge.py`.
	3. Set environment variables under the Render service settings (see ENV below).

- Docker (recommended for ML workloads): build and push Docker image, or deploy to Render/Fly/Heroku using the provided `Dockerfile` or `Procfile`.
- Vercel: Not recommended for heavy ML workloads. If you want serverless-only, consider splitting frontend (Vercel) and ML API (Render/Fly).

Environment variables (recommended)

Set the following environment variables in Render (or your host) instead of committing secrets:

- `TTN_BROKER` (default: eu1.cloud.thethings.network)
- `TTN_PORT` (default: 1883)
- `TTN_USERNAME` (default: bd-test-app2@ttn)
- `TTN_API_KEY` (TTN API key — required for historical fetch and MQTT auth)
- `TTN_DEVICE_ID` (device id, default in repo)
- `THINGSPEAK_API_KEY` (ThingSpeak API key)
- `THINGSPEAK_URL` (optional override)
- `THINGSPEAK_INTERVAL` (seconds between ThingSpeak posts)

Storage notes

- `message_history.json` and `ml_models.pkl` currently live on the instance filesystem. For persistence across deploys/scales, use object storage (S3) and update the code to read/write from S3. I can add S3 support if you want.

If you want, I can:
- Move secrets to environment variables (done for the MQTT bridge already).
- Create Render YAML or help connect your GitHub repository to Render (I added `render.yaml` as an example).
- Deploy the app to Render and verify endpoints if you provide Render/GitHub access.

If you want, I can:
- Move secrets to environment variables.
- Create a `vercel.json` and serverless wrappers for inference endpoints (inference-only) for Vercel deployment.
- Deploy the app to Render (I can create a small guide and necessary files).
