Render deployment steps for your Flask + ML backend

Overview
- This repository already contains a working `Dockerfile`, `requirements.txt`, `Procfile`, and `render.yaml` example that create two services on Render: a web service (Flask app) and a worker (MQTT bridge).
- Recommended approach: use the existing `render.yaml` (update the repo field or configure via Render UI) and set required environment variables in the Render dashboard.

Pre-reqs
1. Push this repository to GitHub (or GitLab). Note the repository URL and branch (e.g., `main`).
2. Create a Render account and link GitHub.

Quick guide

1) Prepare `render.yaml`
- Open `render.yaml` and replace `<REPLACE_WITH_YOUR_GITHUB_REPO>` with your GitHub repo URL (or leave blank and configure within Render UI).

2) Connect repo to Render
- On Render dashboard, click "New" → "Import from GitHub".
- Select your repo and the branch you want to deploy (e.g., `main`).

3) Deploy Web Service
- If you use `render.yaml`, Render will detect and create the web service named `backend-web` using Docker.
- If creating manually, choose:
  - Environment: Docker
  - Build Command: leave empty
  - Start Command: `gunicorn main:app -b 0.0.0.0:$PORT --workers 2`

4) Deploy Worker (MQTT bridge)
- Create a new Worker service.
- Command: `python services/mqtt_bridge.py`
- Ensure the worker uses the same branch and repo.

5) Environment variables (IMPORTANT)
- Set these in Render service's "Environment" settings (both Web and Worker if needed):
  - `TTN_USERNAME` — MQTT username for TTN (or your TTN application id)
  - `TTN_PASSWORD` — MQTT password / access key
  - `THINGSPEAK_KEY` — ThingSpeak write key (if used)
  - `FLASK_ENV` — (optional) `production` or `development`
  - Any other API keys your code relies on

6) Storage and files
- The app writes `message_history.json` and `ml_models.pkl` to the container's filesystem. If you need persistence across deploys, use an external storage (S3) or a managed DB. For prototyping, container-local storage is acceptable.

7) Logs and troubleshooting
- Use Render's Logs view for both services to see stdout/stderr.
- Key things to check in logs when deploying:
  - Successful pip install of `scikit-learn` and `pandas`.
  - `✅ Models loaded successfully from ...` or `📝 No saved models found` messages from `ml_backend.py`.
  - MQTT bridge connecting: `✅ Connected to TTN MQTT broker!` messages.

8) Health check
- After deployment, hit `https://<your-service>.onrender.com/api/health` to verify the app is running and models are loaded.

Optional: Use Render's `Persistent Disk` or S3 for `ml_models.pkl`
- If you need models or message history persisted across redeploys, either:
  - Push model files into the repo (not recommended for secrets/binary size), or
  - Use S3 and update `ml_backend.py`/`mqtt_bridge.py` to read/write from S3, or
  - Use Render's managed disk (paid plans) and mount it to both Web and Worker services.

That's it — if you want, I can:
- Update `render.yaml` with your exact GitHub repository string.
- Add a `render` folder with a CI config file or a `docker-compose.yml` for local testing.
- Add S3 read/write helpers and example env vars for persistence.

If you give me your GitHub repo URL (or say "I want you to patch render.yaml with my repo name"), I'll update `render.yaml` and push the small change here.