from flask import request, jsonify, Blueprint
from helpers.db_connection import pool

floodwatch_bp = Blueprint("floodwatch_bp", __name__, url_prefix="/node")


@floodwatch_bp.route("/locations", methods=["GET"])
def get_node_locations():
    """
    For FloodWatch app:
    GET /node/locations?site_id=1   (site_id optional)

    Returns: list of nodes with node_name, site_name, latitude, longitude,
             and latest temperature + humidity (if any).
    """
    site_id = request.args.get("site_id", type=int)

    conn = pool.get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        sql = """
            SELECT
                nn.node_id,
                nn.node_name,
                s.site_id,
                s.site_name,
                ni.latitude,
                ni.longitude,
                rr.temperature,
                rr.humidity,
                rr.timestamp AS last_timestamp
            FROM tbl_node_identity AS ni
            JOIN tbl_node_name  AS nn ON ni.node_id = nn.node_id
            JOIN tbl_site       AS s  ON ni.site_id = s.site_id
            LEFT JOIN (
                -- latest raw_reading per node/site
                SELECT r.*
                FROM tbl_raw_reading r
                JOIN (
                    SELECT node_id, site_id, MAX(timestamp) AS max_ts
                    FROM tbl_raw_reading
                    GROUP BY node_id, site_id
                ) latest
                ON latest.node_id = r.node_id
                AND latest.site_id = r.site_id
                AND latest.max_ts = r.timestamp
            ) AS rr
              ON rr.node_id = ni.node_id
             AND rr.site_id = ni.site_id
            WHERE ni.isactive = 1
        """

        params = []
        if site_id is not None:
            sql += " AND s.site_id = %s"
            params.append(site_id)

        cur.execute(sql, params)
        rows = cur.fetchall()

        return jsonify({"nodes": rows}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()

@floodwatch_bp.route("/<node_name>/info", methods=["GET"])
def get_node_info(node_name):
    conn = pool.get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        sql = """
            SELECT
                nn.node_id,
                nn.node_name,
                s.site_id,
                s.site_name,
                ni.latitude,
                ni.longitude
            FROM tbl_node_identity AS ni
            JOIN tbl_node_name  AS nn ON ni.node_id = nn.node_id
            JOIN tbl_site       AS s  ON ni.site_id = s.site_id
            WHERE ni.isactive = 1
              AND nn.node_name = %s
        """
        cur.execute(sql, (node_name,))
        row = cur.fetchone()
        if not row:
            return jsonify({"error": "Node not found"}), 404
        return jsonify(row), 200
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()

# ─────────────────────────────────────────────
# NEW ENDPOINT: LATEST PREDICTION FOR A NODE
# ─────────────────────────────────────────────

@floodwatch_bp.route("/<node_name>/prediction", methods=["GET"])
def get_node_prediction(node_name):
    """
    GET /node/<node_name>/prediction

    Returns the most recent predicted temperature for this node from
    tbl_predicted_and_timestamp.

    Response 200:
    {
      "node_name": "node_1",
      "site_id": 1,
      "predicted_temperature": 29.5,
      "predicted_timestamp": "2025-11-16T02:11:00"
    }

    404 if no prediction found.
    """
    conn = pool.get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        sql = """
            SELECT
                node_name,
                site_id,
                predicted_temperature,
                predicted_timestamp
            FROM tbl_predicted_and_timestamp
            WHERE node_name = %s
            ORDER BY predicted_timestamp DESC
            LIMIT 1
        """
        cur.execute(sql, (node_name,))
        row = cur.fetchone()

        if not row:
            return jsonify({"error": "No prediction found for this node"}), 404

        # Ensure datetime is JSON-friendly
        ts = row.get("predicted_timestamp")
        if ts is not None:
            row["predicted_timestamp"] = ts.isoformat()

        return jsonify(row), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()
