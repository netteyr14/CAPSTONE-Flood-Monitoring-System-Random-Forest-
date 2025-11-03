from flask import request, jsonify, Blueprint
from helpers.db_connection import pool

nodes_bp = Blueprint("nodes_bp", __name__, url_prefix="/node")

@nodes_bp.route("/<node_name>/insert_queue", methods=["POST"])
def insert(node_name):
    json_data = request.get_json(silent=True)
    if not json_data:
        return jsonify({"error": "JSON body required"}), 400

    # validate required fields
    if "temperature" not in json_data or "humidity" not in json_data:
        return jsonify({"error": "Provide 'temperature' and 'humidity'"}), 400

    try:
        temperature = float(json_data["temperature"])
        humidity = float(json_data["humidity"])
        node_num = int(json_data.get("node_num", 1))  # default to 1 if missing
    except ValueError:
        return jsonify({"error": "temperature, humidity, and node_num must be numeric"}), 400

    conn = pool.get_connection()
    cursor = conn.cursor(dictionary=True)

    # map node name to column dynamically (e.g., node_1, node_2)
    node_column = node_name.lower()
    valid_nodes = ["node_1", "node_2", "node_3", "node_4"]  # extendable list

    if node_column not in valid_nodes:
        cursor.close()
        conn.close()
        return jsonify({"error": f"Invalid node name '{node_name}'. Must be one of {valid_nodes}."}), 400

    # Build insert query dynamically
    insert_sql = f"""
        INSERT INTO dht11_random_forest (timestamp, temperature, humidity, {node_column}, node_name)
        VALUES (NOW(3), %s, %s, %s, %s)
    """
    cursor.execute(insert_sql, (temperature, humidity, node_num, node_column))
    new_id = cursor.lastrowid

    # get the timestamp of the newly inserted row
    cursor.execute(f"""
        SELECT timestamp FROM dht11_random_forest
        WHERE id=%s AND {node_column}=%s
    """, (new_id, node_num))
    ts_row = cursor.fetchone()
    ts = ts_row["timestamp"] if ts_row else None

    # enqueue to prediction queue
    cursor.execute("""
        INSERT IGNORE INTO queue_table (node_name, ts)
        VALUES (%s, %s)
    """, (node_column, ts))

    # optional: count total rows
    cursor.execute("SELECT COUNT(*) AS cnt FROM dht11_random_forest")
    total_rows = int(cursor.fetchone()["cnt"])

    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({
        "message": "created",
        "id": new_id,
        "node": node_column,
        "timestamp": str(ts),
        "current_row_count": total_rows
    }), 201
