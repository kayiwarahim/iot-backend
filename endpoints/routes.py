import os
from flask import Blueprint, jsonify, send_file


api_bp = Blueprint("api", __name__)


@api_bp.get("/api/data")
def get_sample_data():
    return jsonify(
        {
            "data": [
                {"id": 1, "name": "Sample Item 1", "value": 100},
                {"id": 2, "name": "Sample Item 2", "value": 200},
                {"id": 3, "name": "Sample Item 3", "value": 300},
            ],
            "total": 3,
            "timestamp": "2024-01-01T00:00:00Z",
        }
    )


@api_bp.get("/api/items/<int:item_id>")
def get_item(item_id: int):
    return jsonify(
        {
            "item": {
                "id": item_id,
                "name": f"Sample Item {item_id}",
                "value": item_id * 100,
            },
            "timestamp": "2024-01-01T00:00:00Z",
        }
    )

@api_bp.route('/message_history.json', methods=['GET'])
def serve_message_history():
    try:
        base = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(base, 'message_history.json')
        if not os.path.exists(path):
            return jsonify({'error': 'message_history.json not found'}), 404
        return send_file(path, mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500