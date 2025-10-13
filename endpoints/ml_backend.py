import json
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify, send_file
from datetime import datetime, timedelta
import pickle
import os
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


ml_bp = Blueprint("ml", __name__)


class IoTMLPredictor:
    def __init__(self, base_dir=None):
        self.temp_model = None
        self.humidity_model = None
        self.motion_model = None
        self.anomaly_model = None
        self.scaler = StandardScaler()
        self.models_loaded = False
        # default data file placed at project root if available
        base = base_dir or os.path.dirname(os.path.dirname(__file__))
        self.data_file = os.path.join(base, 'message_history.json')
        self.models_path = os.path.join(base, 'ml_models.pkl')
        # attempt to load previously saved models
        self.load_models(self.models_path)

    def load_historical_data(self, data_file=None):
        data_file = data_file or self.data_file
        try:
            if not os.path.exists(data_file):
                print(f"❌ Data file {data_file} not found")
                return None

            records = []
            with open(data_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line.strip())
                            payload = entry.get('uplink_message', {}).get('decoded_payload', {})
                            ts = entry.get('received_at')
                            if not payload or not ts:
                                continue
                            timestamp = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                            records.append({
                                'timestamp': timestamp,
                                'temperature': float(payload.get('field5', 0)),
                                'humidity': float(payload.get('field3', 0)),
                                'battery': float(payload.get('field1', 0)),
                                'motion': int(payload.get('field4', 0)),
                                'hour': timestamp.hour,
                                'day_of_week': timestamp.weekday(),
                                'month': timestamp.month
                            })
                        except Exception as e:
                            print(f"⚠️ Error parsing line: {e}")
                            continue

            if not records:
                return None

            df = pd.DataFrame(records).sort_values('timestamp')

            # feature engineering
            df['temp_lag1'] = df['temperature'].shift(1)
            df['humidity_lag1'] = df['humidity'].shift(1)
            df['motion_lag1'] = df['motion'].shift(1)
            df['temp_diff'] = df['temperature'].diff()
            df['humidity_diff'] = df['humidity'].diff()

            # cyclical (seasonal) encodings for time
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

            df = df.dropna()
            return df

        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            return None

    def train_models(self, df=None):
        try:
            if df is None:
                df = self.load_historical_data()

            if df is None or len(df) < 10:
                print("❌ Not enough data to train models")
                return False

            feature_cols = [
                'hour', 'day_of_week', 'month', 'battery',
                'temp_lag1', 'humidity_lag1', 'motion_lag1',
                'temp_diff', 'humidity_diff',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos'
            ]

            X = df[feature_cols]
            y_temp = df['temperature']
            y_humidity = df['humidity']

            self.temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.temp_model.fit(X, y_temp)

            self.humidity_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.humidity_model.fit(X, y_humidity)

            from sklearn.ensemble import RandomForestClassifier
            motion_threshold = df['motion'].median()
            y_motion = (df['motion'] > motion_threshold).astype(int)
            self.motion_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.motion_model.fit(X, y_motion)

            self.anomaly_model = IsolationForest(contamination=0.1, random_state=42)
            self.anomaly_model.fit(X)

            self.models_loaded = True

            try:
                temp_pred = self.temp_model.predict(X)
                humidity_pred = self.humidity_model.predict(X)
                from sklearn.metrics import r2_score
                print(f"Temperature R²: {r2_score(y_temp, temp_pred):.3f}")
                print(f"Humidity R²: {r2_score(y_humidity, humidity_pred):.3f}")
            except Exception:
                pass

            print("✅ Models trained successfully")
            return True

        except Exception as e:
            print(f"❌ Error training models: {e}")
            return False

    def predict_next_values(self, current_data):
        if not self.models_loaded:
            return None
        try:
            ts_str = current_data.get('targetTime')
            when = None
            if ts_str:
                try:
                    when = datetime.fromisoformat(ts_str if 'T' in ts_str else ts_str + ':00')
                except Exception:
                    when = datetime.now()
            else:
                when = datetime.now()

            features = np.array([[
                when.hour,
                when.weekday(),
                when.month,
                current_data.get('battery', 3.0),
                current_data.get('temperature', 25.0),
                current_data.get('humidity', 65.0),
                current_data.get('motion', 1000),
                0,
                0,
                np.sin(2 * np.pi * (when.hour / 24.0)),
                np.cos(2 * np.pi * (when.hour / 24.0)),
                np.sin(2 * np.pi * (when.weekday() / 7.0)),
                np.cos(2 * np.pi * (when.weekday() / 7.0)),
                np.sin(2 * np.pi * (when.month / 12.0)),
                np.cos(2 * np.pi * (when.month / 12.0))
            ]])

            temp_pred = self.temp_model.predict(features)[0]
            humidity_pred = self.humidity_model.predict(features)[0]

            motion_pred_prob = self.motion_model.predict_proba(features)[0]
            motion_pred = 1 if motion_pred_prob[1] > 0.5 else 0

            anomaly_score = self.anomaly_model.decision_function(features)[0]
            is_anomaly = anomaly_score < -0.1

            return {
                'temperature_prediction': round(float(temp_pred), 2),
                'humidity_prediction': round(float(humidity_pred), 2),
                'motion_prediction': 'High' if motion_pred == 1 else 'Low',
                'motion_probability': round(float(motion_pred_prob[1]) * 100, 1),
                'anomaly_score': round(float(anomaly_score), 3),
                'is_anomaly': bool(is_anomaly),
                'confidence': {
                    'temperature': round(abs(temp_pred - current_data.get('temperature', temp_pred)), 2),
                    'humidity': round(abs(humidity_pred - current_data.get('humidity', humidity_pred)), 2)
                }
            }

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None

    def predict_trend(self, hours=24):
        if not self.models_loaded:
            return None
        try:
            df = self.load_historical_data()
            last_row = df.iloc[-1] if df is not None and len(df) > 0 else None
            predictions = []
            current_time = datetime.now()

            last_temp = float(last_row['temperature']) if last_row is not None else 25.0
            last_hum = float(last_row['humidity']) if last_row is not None else 65.0
            last_motion = float(last_row['motion']) if last_row is not None else 1000.0
            last_batt = float(last_row['battery']) if last_row is not None else 3.0

            for i in range(hours):
                future_time = current_time + timedelta(hours=i + 1)

                temp_diff = 0.0
                hum_diff = 0.0

                features = np.array([[
                    future_time.hour,
                    future_time.weekday(),
                    future_time.month,
                    last_batt,
                    last_temp,
                    last_hum,
                    last_motion,
                    temp_diff,
                    hum_diff,
                    np.sin(2 * np.pi * (future_time.hour / 24.0)),
                    np.cos(2 * np.pi * (future_time.hour / 24.0)),
                    np.sin(2 * np.pi * (future_time.weekday() / 7.0)),
                    np.cos(2 * np.pi * (future_time.weekday() / 7.0)),
                    np.sin(2 * np.pi * (future_time.month / 12.0)),
                    np.cos(2 * np.pi * (future_time.month / 12.0))
                ]])

                temp_pred = float(self.temp_model.predict(features)[0])
                hum_pred = float(self.humidity_model.predict(features)[0])
                motion_prob = self.motion_model.predict_proba(features)[0]

                predictions.append({
                    'timestamp': future_time.isoformat(),
                    'temperature': round(temp_pred, 2),
                    'humidity': round(hum_pred, 2),
                    'motion_probability': round(float(motion_prob[1]) * 100, 1)
                })

                temp_diff = temp_pred - last_temp
                hum_diff = hum_pred - last_hum
                last_temp = temp_pred
                last_hum = hum_pred
                last_motion = 1_000.0 + 1_000.0 * motion_prob[1]

            return predictions

        except Exception as e:
            print(f"❌ Error predicting trends: {e}")
            return None

    def detect_anomalies(self, current_data):
        if not self.models_loaded:
            return None
        try:
            now = datetime.now()
            features = np.array([[
                now.hour,
                now.weekday(),
                now.month,
                current_data.get('battery', 3.0),
                current_data.get('temperature', 25.0),
                current_data.get('humidity', 65.0),
                current_data.get('motion', 1000),
                0, 0,
                np.sin(2 * np.pi * (now.hour / 24.0)),
                np.cos(2 * np.pi * (now.hour / 24.0)),
                np.sin(2 * np.pi * (now.weekday() / 7.0)),
                np.cos(2 * np.pi * (now.weekday() / 7.0)),
                np.sin(2 * np.pi * (now.month / 12.0)),
                np.cos(2 * np.pi * (now.month / 12.0))
            ]])

            score = self.anomaly_model.decision_function(features)[0]
            return {
                'anomaly_score': round(float(score), 3),
                'is_anomaly': score < -0.1,
                'severity': 'High' if score < -0.5 else 'Medium' if score < -0.1 else 'Low'
            }
        except Exception as e:
            print(f"❌ Anomaly detection error: {e}")
            return None

    def save_models(self, path=None):
        path = path or self.models_path
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'temp_model': self.temp_model,
                    'humidity_model': self.humidity_model,
                    'motion_model': self.motion_model,
                    'anomaly_model': self.anomaly_model,
                    'scaler': self.scaler
                }, f)
            print("✅ Models saved successfully")
        except Exception as e:
            print(f"❌ Error saving models: {e}")

    def load_models(self, path=None):
        # Check multiple candidate locations for the model file
        candidates = []
        if path:
            candidates.append(path)
        candidates.append(self.models_path)
        # also check inside the endpoints directory where the file was discovered
        endpoints_possible = os.path.join(os.path.dirname(__file__), 'ml_models.pkl')
        candidates.append(endpoints_possible)
        # project-level and a 'models' subfolder
        candidates.append(os.path.join(os.path.dirname(self.models_path), 'models', 'ml_models.pkl'))

        tried = []
        try:
            for p in candidates:
                if not p:
                    continue
                tried.append(p)
                if os.path.exists(p):
                    with open(p, 'rb') as f:
                        data = pickle.load(f)
                    self.temp_model = data.get('temp_model')
                    self.humidity_model = data.get('humidity_model')
                    self.motion_model = data.get('motion_model')
                    self.anomaly_model = data.get('anomaly_model')
                    self.scaler = data.get('scaler', StandardScaler())
                    self.models_loaded = all([self.temp_model, self.humidity_model, self.motion_model, self.anomaly_model])
                    print(f"✅ Models loaded successfully from {p}")
                    return

            # if we reach here no file was found
            print("📝 No saved models found in candidates (checked: ")
            for c in tried:
                print(f"  - {c}")
        except Exception as e:
            print(f"❌ Error loading models from candidates {tried}: {e}")


# Initialize predictor with project root as base
predictor = IoTMLPredictor()


@ml_bp.route('/api/health', methods=['GET'])
def health_check():
    df = predictor.load_historical_data()
    return jsonify({
        'status': 'healthy',
        'models_loaded': predictor.models_loaded,
        'historical_data_points': len(df) if df is not None else 0,
        'timestamp': datetime.now().isoformat()
    })


@ml_bp.route('/api/train', methods=['POST'])
def train_models_endpoint():
    df = predictor.load_historical_data()
    if df is None or len(df) < 10:
        return jsonify({'error': 'Insufficient data for training'}), 400
    success = predictor.train_models(df)
    if success:
        predictor.save_models()
        return jsonify({'message': 'Models trained successfully', 'data_points': len(df)})
    else:
        return jsonify({'error': 'Training failed'}), 500


@ml_bp.route('/api/retrain', methods=['POST'])
def retrain_models():
    try:
        df = predictor.load_historical_data()
        success = predictor.train_models(df)
        if success:
            predictor.save_models()
            return jsonify({'status': 'success', 'message': 'Models retrained successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to train models'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@ml_bp.route('/api/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        if not predictor.models_loaded:
            return jsonify({'error': 'Models not trained yet'}), 400
        preds = predictor.predict_next_values(data)
        if preds is None:
            return jsonify({'error': 'Prediction failed'}), 500
        return jsonify(preds)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ml_bp.route('/message_history.json', methods=['GET'])
def serve_message_history():
    """Serve message_history.json. Prefer the dashboard public file written by the bridge
    (Dashboard/IoTDashboard/public/message_history.json) and fall back to project root.
    """
    try:
        base = os.path.dirname(os.path.dirname(__file__))
        candidates = [
            os.path.join(base, 'Dashboard', 'IoTDashboard', 'public', 'message_history.json'),
            os.path.join(base, 'message_history.json'),
            os.path.join(base, 'public', 'message_history.json'),
        ]
        for p in candidates:
            if os.path.exists(p):
                return send_file(p, mimetype='application/json')
        return jsonify({'error': 'message_history.json not found in known locations', 'checked': candidates}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ml_bp.route('/latest.json', methods=['GET'])
def serve_latest_json():
    try:
        base = os.path.dirname(os.path.dirname(__file__))
        candidates = [
            os.path.join(base, 'Dashboard', 'public', 'latest.json'),
            os.path.join(base, 'latest.json'),
        ]
        for p in candidates:
            if os.path.exists(p):
                return send_file(p, mimetype='application/json')
        return jsonify({'error': 'latest.json not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ml_bp.route('/api/trends', methods=['GET'])
def trends_endpoint():
    try:
        hours = int(request.args.get('hours', 24))
        if not predictor.models_loaded:
            return jsonify({'error': 'Models not trained yet'}), 400
        trends = predictor.predict_trend(hours)
        if trends is None:
            return jsonify({'error': 'Trend prediction failed'}), 500
        return jsonify({'trends': trends})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ml_bp.route('/api/anomalies', methods=['POST'])
def anomalies_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        if not predictor.models_loaded:
            return jsonify({'error': 'Models not trained yet'}), 400
        res = predictor.detect_anomalies(data)
        if res is None:
            return jsonify({'error': 'Anomaly detection failed'}), 500
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
