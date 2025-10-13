import json
import time
import requests
import threading
from queue import Queue
from datetime import datetime
import os
import shutil
from pathlib import Path

msg_queue = Queue()

# Configuration (defaults are provided but it's strongly recommended to set env vars in Render)
import os as _os
broker = _os.getenv('TTN_BROKER', 'eu1.cloud.thethings.network')
port = int(_os.getenv('TTN_PORT', '1883'))
username = _os.getenv('TTN_USERNAME', 'bd-test-app2@ttn')
password = _os.getenv('TTN_API_KEY', 'NNSXS.NGFSXX4UXDX55XRIDQZS6LPR4OJXKIIGSZS56CQ.6O4WUAUHFUAHSTEYRWJX6DDO7TL2IBLC7EV2LS4EHWZOOEPCEUOA')
device_id = _os.getenv('TTN_DEVICE_ID', 'lht65n-01-temp-humidity-sensor')

# ThingSpeak settings (override via env vars)
THINGSPEAK_API_KEY = _os.getenv('THINGSPEAK_API_KEY', 'MXE9E1DQ3RFXZPEB')
THINGSPEAK_URL = _os.getenv('THINGSPEAK_URL', 'https://api.thingspeak.com/update')
THINGSPEAK_INTERVAL = int(_os.getenv('THINGSPEAK_INTERVAL', '15'))  # seconds between updates

# Paths (adapted so this module can be imported)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = PROJECT_ROOT / "Dashboard" / "IoTDashboard" / "public"
FRONTEND_JSON_FILE = PUBLIC_DIR / "message_history.json"
ROOT_ARCHIVE_FILE = PROJECT_ROOT / "message_history_root_archive.jsonl"
LATEST_FILE = PUBLIC_DIR / "latest.json"
LEGACY_ROOT_FILE = PROJECT_ROOT / "message_history.json"
JSON_FILE = str(FRONTEND_JSON_FILE)

def _ensure_public_file():
    if not PUBLIC_DIR.exists():
        try:
            PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
            print(f"📂 Created public directory: {PUBLIC_DIR}")
        except Exception as e:
            print(f"❌ Cannot create public directory {PUBLIC_DIR}: {e}")
            return False
    return True

def _reconcile_legacy_file():
    try:
        if not LEGACY_ROOT_FILE.exists():
            return
        legacy_stat = LEGACY_ROOT_FILE.stat()
        needs_copy = False
        if not FRONTEND_JSON_FILE.exists():
            needs_copy = True
        else:
            pub_stat = FRONTEND_JSON_FILE.stat()
            if legacy_stat.st_mtime > pub_stat.st_mtime + 1:
                needs_copy = True
            elif legacy_stat.st_size > pub_stat.st_size and legacy_stat.st_mtime >= pub_stat.st_mtime:
                needs_copy = True
        if needs_copy:
            if _ensure_public_file():
                shutil.copy2(LEGACY_ROOT_FILE, FRONTEND_JSON_FILE)
                print(f"🔄 Reconciled legacy file → copied {LEGACY_ROOT_FILE.name} to public directory")
    except Exception as e:
        print(f"⚠️ Legacy reconciliation failed: {e}")

def _archive_line(line: str):
    try:
        with open(ROOT_ARCHIVE_FILE, 'a') as af:
            af.write(line)
    except Exception as e:
        print(f"⚠️ Could not write archive line: {e}")

def _write_lines(lines):
    if not _ensure_public_file():
        return False
    try:
        with open(JSON_FILE, 'w') as jf:
            jf.writelines(lines)
        print(f"🪞 Wrote {len(lines)} lines to {JSON_FILE}")
        try:
            with open(LEGACY_ROOT_FILE, 'w') as lf:
                lf.writelines(lines)
        except Exception as e:
            print(f"⚠️ Could not write legacy root file: {e}")
    except Exception as e:
        print(f"❌ Failed writing canonical JSON file: {e}")
    for line in lines:
        _archive_line(line)

def _append_line(line: str):
    if _ensure_public_file():
        try:
            with open(JSON_FILE, 'a') as jf:
                jf.write(line)
            try:
                with open(LEGACY_ROOT_FILE, 'a') as lf:
                    lf.write(line)
            except Exception as e:
                print(f"⚠️ Could not append to legacy root file: {e}")
        except Exception as e:
            print(f"❌ Failed appending to canonical file: {e}")
    _archive_line(line)

def _write_latest(payload: dict):
    try:
        if not _ensure_public_file():
            return
        uplink = payload.get("uplink_message", {})
        decoded = uplink.get("decoded_payload", {})
        latest_obj = {
            "received_at": payload.get("received_at", uplink.get("received_at")),
            "temperature": decoded.get("field5"),
            "humidity": decoded.get("field3"),
            "battery": decoded.get("field1"),
            "motion": decoded.get("field4"),
            "work_mode": decoded.get("Work_mode") or decoded.get("work_mode"),
            "exti": decoded.get("Exti_pin_level") or decoded.get("exti_pin_level"),
        }
        with open(LATEST_FILE, 'w') as lf:
            json.dump(latest_obj, lf)
    except Exception as e:
        print(f"⚠️ Could not update latest.json: {e}")

def get_historical_sensor_data():
    last_fetch_file = PROJECT_ROOT / "last_historical_fetch.txt"
    if last_fetch_file.exists():
        try:
            with open(last_fetch_file, 'r') as f:
                last_fetch_time = float(f.read().strip())
            time_since_last_fetch = time.time() - last_fetch_time
            if time_since_last_fetch < 3600:
                print(f"⏭️ Skipping historical data fetch (last fetch was {time_since_last_fetch/60:.1f} minutes ago)")
                return True
        except Exception as e:
            print(f"⚠️ Could not read last fetch time: {e}")

    app_id = "bd-test-app2"
    api_key = password  # original used same api key variable value
    url = f"https://{broker}/api/v3/as/applications/{app_id}/devices/{device_id}/packages/storage/uplink_message"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"last": "12h"}

    print("🔄 Fetching 12-hour historical data from TTN...")
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        response_text = response.text.strip()
        print(f"✅ Historical data fetched successfully!")
        try:
            lines = response_text.split('\n')
            messages = []
            for line in lines:
                if line.strip():
                    message = json.loads(line)
                    if 'result' in message:
                        messages.append(message['result'])

            if messages:
                existing_data = []
                if os.path.exists(JSON_FILE):
                    try:
                        with open(JSON_FILE, 'r') as f:
                            existing_lines = f.readlines()
                            for line in existing_lines:
                                if line.strip():
                                    existing_data.append(json.loads(line.strip()))
                        print(f"📁 Found {len(existing_data)} existing entries in {JSON_FILE}")
                    except Exception as e:
                        print(f"⚠️ Could not read existing data: {e}")

                all_messages = existing_data + messages
                seen_timestamps = set()
                unique_messages = []
                for msg in all_messages:
                    timestamp = msg.get('received_at', '')
                    if timestamp and timestamp not in seen_timestamps:
                        seen_timestamps.add(timestamp)
                        unique_messages.append(msg)

                unique_messages.sort(key=lambda x: x.get('received_at', ''))
                buffer_lines = [json.dumps(message) + '\n' for message in unique_messages]
                _write_lines(buffer_lines)
                if unique_messages:
                    try:
                        _write_latest(unique_messages[-1])
                    except Exception as e:
                        print(f"⚠️ Could not write latest from historical set: {e}")

                print(f"💾 Combined data saved to {JSON_FILE} ({len(unique_messages)} total entries)")
                with open(last_fetch_file, 'w') as f:
                    f.write(str(time.time()))
                return True
            else:
                print("⚠️ No historical data found in response")
                return False
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing historical data: {e}")
            return False
    else:
        print(f"❌ Error fetching historical data: {response.status_code} - {response.text}")
        return False

def save_to_json(payload):
    try:
        line = json.dumps(payload) + '\n'
        _append_line(line)
        _write_latest(payload)
        _reconcile_legacy_file()
        with open(JSON_FILE, 'r') as f:
            line_count = sum(1 for _ in f)
        if line_count > 1000:
            with open(JSON_FILE, 'r') as f:
                lines = f.readlines()
            lines = lines[-1000:]
            _write_lines(lines)
            line_count = 1000
        print(f"💾 Saved to {JSON_FILE} (total entries: {line_count})")
    except Exception as e:
        print(f"❌ Error saving to JSON: {e}")

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("✅ Connected to TTN MQTT broker!")
        client.subscribe(f"v3/{username}/devices/{device_id}/up")
    else:
        print(f"❌ Failed to connect, return code {rc}")
        time.sleep(5*60)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
    except Exception:
        return
    msg_queue.put(payload)
    save_to_json(payload)
    print("📥 New message added to queue and saved to JSON")

def thingspeak_worker():
    last_post_time = 0
    while True:
        payload = msg_queue.get()
        decoded = payload.get("uplink_message", {}).get("decoded_payload", {})
        battery = decoded.get("field1")
        humidity = decoded.get("field3")
        motion = decoded.get("field4")
        temperature = decoded.get("field5")

        received_at = (
            payload.get("received_at")
            or payload.get("uplink_message", {}).get("received_at")
            or payload.get("uplink_message", {}).get("rx_metadata", [{}])[0].get("time")
        )
        if received_at:
            try:
                iso_str = received_at.replace("Z", "+00:00")
                dt_obj = datetime.fromisoformat(iso_str)
                dt_local = dt_obj.astimezone()
                received_str = dt_local.strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                received_str = received_at
        else:
            received_str = time.strftime("%Y-%m-%d %H:%M:%S %Z")

        now = time.time()
        wait_time = THINGSPEAK_INTERVAL - (now - last_post_time)
        if wait_time > 0:
            time.sleep(wait_time)

        data = {
            "api_key": THINGSPEAK_API_KEY,
            "field1": battery,
            "field2": temperature,
            "field3": humidity,
            "field4": motion
        }
        try:
            response = requests.post(THINGSPEAK_URL, data=data)
            if response.status_code == 200 and response.text.strip() != "0":
                print(f"✅ Data sent @ {received_str} → Temp={temperature}, Humidity={humidity}, Battery={battery}, Motion={motion}")
                last_post_time = time.time()
            else:
                print("❌ ThingSpeak update failed:", response.text)
        except Exception as e:
            print("❌ Error sending to ThingSpeak:", e)

        msg_queue.task_done()

def run_bridge():
    print("🚀 Starting TTN Bridge Service (public file is canonical)...")
    print(f"   Public JSON file path: {FRONTEND_JSON_FILE}")
    print(f"   Legacy root file path: {LEGACY_ROOT_FILE}")
    print(f"   Latest JSON path: {LATEST_FILE}")
    if not _ensure_public_file():
        print("❌ Public directory missing; dashboard will not update until it exists.")
    else:
        _reconcile_legacy_file()
    get_historical_sensor_data()

    # start thingspeak worker
    threading.Thread(target=thingspeak_worker, daemon=True).start()

    # set up mqtt client lazily to avoid mandatory dependency at import time
    try:
        import paho.mqtt.client as mqtt
    except Exception as e:
        print(f"⚠️ paho-mqtt not installed or failed to import: {e}")
        return

    client = mqtt.Client()
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(broker, port, 60)
        client.loop_forever()
    except Exception as e:
        print(f"❌ MQTT client failed: {e}")

def start_in_background():
    threading.Thread(target=run_bridge, daemon=True).start()


if __name__ == '__main__':
    # allow running this module directly as a background worker
    run_bridge()
