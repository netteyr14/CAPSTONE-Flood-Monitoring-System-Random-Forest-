from flask import request, jsonify, Blueprint
from helpers.db_connection import pool

nodes_bp = Blueprint("nodes_bp", __name__, url_prefix="/node")

@nodes_bp.route("/<node_name>/insert_queue", methods=["POST"])
def insert(node_name):
    # Parse JSON
    json_data = request.get_json(silent=True)
    if not json_data:
        return jsonify({"error": "JSON body required"}), 400

    # Validate required fields
    if "temperature" not in json_data or "humidity" not in json_data:
        return jsonify({"error": "Provide 'temperature' and 'humidity'"}), 400

    try:
        temperature = float(json_data["temperature"])
        humidity = float(json_data["humidity"])
        node_id = int(json_data.get("node_num", 1))
        site_id = int(json_data.get("site_num", 1))
    except ValueError:
        return jsonify({"error": "temperature, humidity, node_num, site_num must be numeric"}), 400

    # Get DB connection
    conn = pool.get_connection()
    cursor = conn.cursor(dictionary=True)

    # Insert reading
    insert_sql = """
        INSERT INTO tbl_raw_reading (timestamp, temperature, humidity, node_id, site_id)
        VALUES (NOW(3), %s, %s, %s, %s)
    """
    cursor.execute(insert_sql, (temperature, humidity, node_id, site_id))
    new_id = cursor.lastrowid

    # Get timestamp of newly inserted row
    cursor.execute("""
        SELECT timestamp FROM tbl_raw_reading
        WHERE id=%s AND node_id=%s AND site_id=%s
    """, (new_id, node_id, site_id))
    ts_row = cursor.fetchone()
    ts = ts_row["timestamp"] if ts_row else None

    # Insert into prediction queue
    cursor.execute("""
        INSERT IGNORE INTO tbl_queue (node_name, ts)
        VALUES (%s, %s)
    """, (node_name, ts))

    # Optional: get total row count
    cursor.execute("SELECT COUNT(*) AS cnt FROM tbl_raw_reading")
    total_rows = int(cursor.fetchone()["cnt"])

    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({
        "message": "created",
        "id": new_id,
        "node_name": node_name,
        "node_id": node_id,
        "site_id": site_id,
        "timestamp": str(ts),
        "current_row_count": total_rows
    }), 201
