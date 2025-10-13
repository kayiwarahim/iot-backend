import paho.mqtt.client as mqtt
import json
import time
import requests
import threading
from queue import Queue
from datetime import datetime, timedelta
import os
import shutil

# Configuration
broker = "eu1.cloud.thethings.network"
port = 1883
username = "bd-test-app2@ttn"
password = "NNSXS.NGFSXX4UXDX55XRIDQZS6LPR4OJXKIIGSZS56CQ.6O4WUAUHFUAHSTEYRWJX6DDO7TL2IBLC7EV2LS4EHWZOOEPCEUOA"
device_id = "lht65n-01-temp-humidity-sensor"

# ThingSpeak settings
THINGSPEAK_API_KEY = "MXE9E1DQ3RFXZPEB"
THINGSPEAK_URL = "https://api.thingspeak.com/update"
THINGSPEAK_INTERVAL = 15  # Minimum seconds between updates

# MQTT topic
topic = f"v3/{username}/devices/{device_id}/up"

# Queue to store messages
msg_queue = Queue()

from pathlib import Path

# JSON file paths
SCRIPT_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = SCRIPT_DIR / "Dashboard" / "IoTDashboard" / "public"
FRONTEND_JSON_FILE = PUBLIC_DIR / "message_history.json"
ROOT_ARCHIVE_FILE = SCRIPT_DIR / "message_history_root_archive.jsonl"
LATEST_FILE = PUBLIC_DIR / "latest.json"
LEGACY_ROOT_FILE = SCRIPT_DIR / "message_history.json"  # old location creating confusion

# Use the public file as the canonical JSONL so the frontend always sees fresh data
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
    """If a legacy root-level message_history.json is newer or larger than the public one, copy it over.

    This handles the situation where the running process still writes to the old path, producing
    fresh data the frontend never sees. We do a simple heuristic:
      - if legacy file exists and (public missing OR legacy mtime > public mtime OR legacy size > public size)
        then copy it to the public location.
    """
    try:
        if not LEGACY_ROOT_FILE.exists():
            return
        legacy_stat = LEGACY_ROOT_FILE.stat()
        needs_copy = False
        if not FRONTEND_JSON_FILE.exists():
            needs_copy = True
        else:
            pub_stat = FRONTEND_JSON_FILE.stat()
            if legacy_stat.st_mtime > pub_stat.st_mtime + 1:  # 1 second tolerance
                needs_copy = True
            elif legacy_stat.st_size > pub_stat.st_size and legacy_stat.st_mtime >= pub_stat.st_mtime:
                needs_copy = True
        if needs_copy:
            if _ensure_public_file():
                shutil.copy2(LEGACY_ROOT_FILE, FRONTEND_JSON_FILE)
                print(f"🔄 Reconciled legacy file → copied {LEGACY_ROOT_FILE.name} ({legacy_stat.st_size} bytes) to public directory")
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
        # Dual write to legacy root for backward compatibility until old process instances are gone
        try:
            with open(LEGACY_ROOT_FILE, 'w') as lf:
                lf.writelines(lines)
        except Exception as e:
            print(f"⚠️ Could not write legacy root file: {e}")
    except Exception as e:
        print(f"❌ Failed writing canonical JSON file: {e}")
    # Append all lines to archive (could be large; optional down-sampling in future)
    for line in lines:
        _archive_line(line)

def _append_line(line: str):
    if _ensure_public_file():
        try:
            with open(JSON_FILE, 'a') as jf:
                jf.write(line)
            # Also append to legacy root for safety
            try:
                with open(LEGACY_ROOT_FILE, 'a') as lf:
                    lf.write(line)
            except Exception as e:
                print(f"⚠️ Could not append to legacy root file: {e}")
        except Exception as e:
            print(f"❌ Failed appending to canonical file: {e}")
    _archive_line(line)

def _write_latest(payload: dict):
    """Write a simplified latest.json for the frontend to poll quickly."""
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

# Fetch Historical Data
def get_historical_sensor_data():
    """Fetch 12 hours of historical sensor data from TTN API"""
    # Check if we should skip historical data fetch (to avoid too frequent API calls)
    last_fetch_file = "last_historical_fetch.txt"
    if os.path.exists(last_fetch_file):
        try:
            with open(last_fetch_file, 'r') as f:
                last_fetch_time = float(f.read().strip())
            time_since_last_fetch = time.time() - last_fetch_time
            if time_since_last_fetch < 3600:  # Skip if less than 1 hour since last fetch
                print(f"⏭️ Skipping historical data fetch (last fetch was {time_since_last_fetch/60:.1f} minutes ago)")
                return True
        except Exception as e:
            print(f"⚠️ Could not read last fetch time: {e}")
    
    app_id = "bd-test-app2"
    api_key = "NNSXS.NGFSXX4UXDX55XRIDQZS6LPR4OJXKIIGSZS56CQ.6O4WUAUHFUAHSTEYRWJX6DDO7TL2IBLC7EV2LS4EHWZOOEPCEUOA"
    url = f"https://{broker}/api/v3/as/applications/{app_id}/devices/{device_id}/packages/storage/uplink_message"
    
    # Set authorization header
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "last": "12h"  # get messages from last 12 hours. Max 48 hours. Possible values: 12m (12 minutes)
    }
    
    print("🔄 Fetching 12-hour historical data from TTN...")
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        response_text = response.text.strip()
        print(f"✅ Historical data fetched successfully!")
        
        # Parse the JSONL response (one JSON object per line)
        try:
            lines = response_text.split('\n')
            messages = []
            for line in lines:
                if line.strip():
                    message = json.loads(line)
                    if 'result' in message:
                        messages.append(message['result'])
            
            if messages:
                print(f"📊 Found {len(messages)} historical messages")
                
                # Check if file already exists and has data
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
                
                # Combine historical and existing data, removing duplicates
                all_messages = existing_data + messages
                
                # Remove duplicates based on received_at timestamp
                seen_timestamps = set()
                unique_messages = []
                for msg in all_messages:
                    timestamp = msg.get('received_at', '')
                    if timestamp and timestamp not in seen_timestamps:
                        seen_timestamps.add(timestamp)
                        unique_messages.append(msg)
                
                # Sort by received_at timestamp
                unique_messages.sort(key=lambda x: x.get('received_at', ''))
                
                # Save all unique messages
                buffer_lines = [json.dumps(message) + '\n' for message in unique_messages]
                _write_lines(buffer_lines)
                # Update latest.json with the newest historical entry
                if unique_messages:
                    try:
                        _write_latest(unique_messages[-1])
                    except Exception as e:
                        print(f"⚠️ Could not write latest from historical set: {e}")
                
                print(f"💾 Combined data saved to {JSON_FILE} ({len(unique_messages)} total entries)")
                
                # Save the fetch timestamp
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
    """Save payload to JSON file (JSONL format - one JSON object per line)"""
    try:
        # Append new payload to file (JSONL format)
        line = json.dumps(payload) + '\n'
        _append_line(line)
        _write_latest(payload)
        # After each append, attempt reconciliation in case another process still writes legacy file
        _reconcile_legacy_file()
        
        # Count total lines in file
        with open(JSON_FILE, 'r') as f:
            line_count = sum(1 for line in f)
        
        # If file gets too large, keep only last 1000 entries
        if line_count > 1000:
            # Read all lines
            with open(JSON_FILE, 'r') as f:
                lines = f.readlines()
            
            # Keep only last 1000 lines
            lines = lines[-1000:]
            
            # Write back to file
            _write_lines(lines)
            
            line_count = 1000
        
        print(f"💾 Saved to {JSON_FILE} (total entries: {line_count})")
        
    except Exception as e:
        print(f"❌ Error saving to JSON: {e}")

# Callback: When connected to broker
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("✅ Connected to TTN MQTT broker!")
        client.subscribe(topic)
    else:
        print(f"❌ Failed to connect, return code {rc}")
        time.sleep(5*60)

# Callback: When a message is received
def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    msg_queue.put(payload)  # Put payload in queue
    
    # Also save to JSON file immediately
    save_to_json(payload)
    print("📥 New message added to queue and saved to JSON")

# Worker to send messages to ThingSpeak respecting rate limit
def thingspeak_worker():
    last_post_time = 0
    while True:
        payload = msg_queue.get()  # Wait for a message
        decoded = payload["uplink_message"]["decoded_payload"]
        battery = decoded.get("field1")
        humidity = decoded.get("field3")
        motion = decoded.get("field4")
        temperature = decoded.get("field5")

        # Extract the original received timestamp from TTN if available
        received_at = (
            payload.get("received_at")
            or payload.get("uplink_message", {}).get("received_at")
            or payload.get("uplink_message", {}).get("rx_metadata", [{}])[0].get("time")
        )
        # Convert to a readable local time string
        if received_at:
            try:
                # Normalize Z suffix for fromisoformat
                iso_str = received_at.replace("Z", "+00:00")
                dt_obj = datetime.fromisoformat(iso_str)
                # Convert to local timezone
                dt_local = dt_obj.astimezone()
                received_str = dt_local.strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                received_str = received_at  # Fallback: raw string
        else:
            received_str = time.strftime("%Y-%m-%d %H:%M:%S %Z")

        # Rate limit
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
                print(
                    f"✅ Data sent @ {received_str} → Temp={temperature}, Humidity={humidity}, Battery={battery}, Motion={motion}"
                )
                last_post_time = time.time()
            else:
                print("❌ ThingSpeak update failed:", response.text)
        except Exception as e:
            print("❌ Error sending to ThingSpeak:", e)

        msg_queue.task_done()  # Mark task done

# Fetch historical data first
print("🚀 Starting TTN Bridge Service (public file is canonical)...")
print(f"   Public JSON file path: {FRONTEND_JSON_FILE}")
print(f"   Legacy root file path: {LEGACY_ROOT_FILE}")
print(f"   Latest JSON path: {LATEST_FILE}")
if not _ensure_public_file():
    print("❌ Public directory missing; dashboard will not update until it exists.")
else:
    _reconcile_legacy_file()
get_historical_sensor_data()

# Start ThingSpeak worker thread
threading.Thread(target=thingspeak_worker, daemon=True).start()

# Set up MQTT client
client = mqtt.Client()
client.username_pw_set(username, password)
client.on_connect = on_connect
client.on_message = on_message

# Connect to broker and start loop
client.connect(broker, port, 60)
client.loop_forever()
