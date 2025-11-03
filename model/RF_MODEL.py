import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from helpers.db_connection import pool

TABLE_SENSOR = "dht11_random_forest"
MIN_REQUIRED_ROWS = 100  # minimum rows to train
SLEEP_IDLE = 0.3
RETRAIN_AFTER = 10

# -----------------------------
# FETCHING DATA
# -----------------------------
def fetch_latest_rows(node_name):
    conn = pool.get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT timestamp, temperature, humidity, node_1, node_2
        FROM dht11_random_forest
        WHERE node_name = %s
        ORDER BY timestamp ASC
    """, (node_name,))
    latest_rows = cursor.fetchall()
    print(f"📦 Rows fetched for {node_name}: {len(latest_rows)}")  # Debug print
    cursor.close()
    conn.close()
    return latest_rows

# -----------------------------
# QUEUE CLAIM FUNCTION
# -----------------------------
def claim_job(conn):
    cur = conn.cursor(dictionary=True)
    cur.execute("START TRANSACTION")
    cur.execute("""
        SELECT node_name, ts FROM queue_table
        WHERE status='queued'
        ORDER BY ts ASC
        LIMIT 1
        FOR UPDATE SKIP LOCKED
    """)
    row = cur.fetchone()
    if not row:
        cur.execute("COMMIT")
        cur.close()
        return None
    cur.execute("""
        UPDATE queue_table
        SET status='processing', attempt=attempt+1
        WHERE node_name=%s AND ts=%s
    """, (row["node_name"], row["ts"]))
    cur.execute("COMMIT")
    cur.close()
    return row

# -----------------------------
# JOB SUCCESS FUNCTION
# -----------------------------
def job_success(conn, node_name, ts):
    """
    Mark a queue entry as successfully processed.
    """
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE queue_table
            SET status='done', completed_at=NOW()
            WHERE node_name=%s AND ts=%s
        """, (node_name, ts))
        conn.commit()
        cur.close()
        print(f"✅ Job success recorded for node '{node_name}' at {ts}")
    except Exception as e:
        print(f"⚠️ Failed to mark job success for {node_name}: {e}")

# -----------------------------
# JOB FAIL FUNCTION
# -----------------------------
def job_fail(conn, node_name, ts, reason=None):
    """
    Mark a queue entry as failed, optionally logging a reason.
    """
    try:
        cur = conn.cursor()
        if reason:
            cur.execute("""
                UPDATE queue_table
                SET status='failed', completed_at=NOW(), fail_reason=%s
                WHERE node_name=%s AND ts=%s
            """, (reason, node_name, ts))
        else:
            cur.execute("""
                UPDATE queue_table
                SET status='failed', completed_at=NOW()
                WHERE node_name=%s AND ts=%s
            """, (node_name, ts))
        conn.commit()
        cur.close()
        print(f"❌ Job failed for node '{node_name}' at {ts}. Reason: {reason or 'Unknown'}")
    except Exception as e:
        print(f"⚠️ Failed to mark job as failed for {node_name}: {e}")

# -----------------------------
# DATATYPE SETTING
# -----------------------------
def clean_dataframe(rows):
    if rows is None or len(rows) == 0:
        empty_df = pd.DataFrame(columns=["temperature", "humidity"])
        empty_df.index = pd.DatetimeIndex([], name="timestamp")
        return empty_df

    df = pd.DataFrame(rows)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp").set_index("timestamp")

    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
    df = df.dropna(subset=["temperature", "humidity"])

    return df

# -----------------------------
# TIME SERIES FIXING -ITO ANG CLEANING
# -----------------------------
def enforce_fixed_interval(df, frequency):
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.resample(frequency).mean()
    df = df.interpolate(method="time")
    df = df.ffill()
    df = df.bfill()
    return df

# -----------------------------
# LAG FEATURE GENERATION
# -----------------------------
def make_lag_features(df, n_lags):
    feat = df.copy()
    for lag_number in range(1, n_lags + 1):
        feat[f"temp_lag{lag_number}"] = feat["temperature"].shift(lag_number)
        feat[f"hum_lag{lag_number}"] = feat["humidity"].shift(lag_number)

    feat["target_next_temp"] = feat["temperature"].shift(-1)
    feat = feat.dropna()
    return feat

# -----------------------------
# MODEL TRAINING
# -----------------------------
def train_model(df):
    if len(df) < MIN_REQUIRED_ROWS:
        print(f"⚠️ Not enough data to train. Need {MIN_REQUIRED_ROWS}, got {len(df)}.")
        return None

    feature_columns = [col for col in df.columns if col.startswith("temp_lag") or col.startswith("hum_lag")]
    X = df[feature_columns]
    y = df["target_next_temp"]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

# Caping trainset
def take_training_window(df, window_size):
    if len(df) > window_size:
        start_index = len(df) - window_size # 800-500=300
        end_index = len(df)#800
        df_window = df.iloc[start_index:end_index]#300:800
        return df_window
    else:
        return df
    
def prepare_predict_input(df, n_lags):
    """
    Build one row of lag-based features (like temp_lag1..n, hum_lag1..n)
    from the most recent readings.
    """
    if len(df) < n_lags:
        print(f"⚠️ Not enough rows ({len(df)}) to build {n_lags}-lag prediction input.")
        return None

    feature_dict = {}
    for lag_number in range(1, n_lags + 1):
        feature_dict[f"temp_lag{lag_number}"] = float(df["temperature"].iloc[-lag_number])
        feature_dict[f"hum_lag{lag_number}"] = float(df["humidity"].iloc[-lag_number])

    return pd.DataFrame([feature_dict])

def predict_next_step(model, df_recent, last_raw_ts, n_lags=3):
    """
    Predict the next temperature based on latest 'n_lags' readings.
    """
    if model is None:
        print("⚠️ No trained model available for prediction.")
        return None, None

    # Prepare lag-based input
    X_pred = prepare_predict_input(df_recent, n_lags)
    if X_pred is None:
        return None, None

    # Perform prediction
    y_pred = model.predict(X_pred)[0]

    # Predict timestamp (next 5 minutes)
    next_timestamp = last_raw_ts + pd.to_timedelta("5min")
    # print(f"🧩 Predicted next temp = {y_pred:.2f}°C at {next_timestamp}")
    return next_timestamp, float(y_pred)

# -----------------------------
# MAIN WORKER LOOP
# -----------------------------
def worker_loop(conn):
    model = None
    made = 0
    idle = SLEEP_IDLE
    cur = conn.cursor(dictionary=True)

    query = """
        SELECT timestamp, temperature, humidity, node_1, node_2
        FROM dht11_random_forest
        ORDER BY timestamp DESC;
    """
    
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()

    df_all = pd.DataFrame(rows)
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
    df_all = df_all.set_index("timestamp")

    df_with_lags = enforce_fixed_interval(df_all, "5min")
    df_with_lags = make_lag_features(df_with_lags, n_lags=3)

    df_with_clean = clean_dataframe(df_with_lags) # ayusin yung datatypes

    df_with_cap = take_training_window(df_with_clean, 500) #caps to 500

    print("Model: RANDOM FOREST: Train sets - ", len(df_with_cap))

    initial_model = train_model(df_with_cap)

    if initial_model is not None:
        print("✅ Model trained successfully.")
    else:
        print("❌ Model training failed.")

    while True:
        job = claim_job(conn)
        if not job:
            print("ℹ️ No job found, sleeping...")
            time.sleep(idle)
            idle = min(2.0, idle * 1.5)
            continue

        idle = SLEEP_IDLE
        node = job["node_name"]
        ts_for_insert_and_queue = pd.to_datetime(job["ts"])
        print(f"🎯 Processing job from node '{node}' at {ts_for_insert_and_queue}")

         # Step 1: Fetch latest raw data again before predicting
        latest_rows = fetch_latest_rows(node)

        df_latest = clean_dataframe(latest_rows) #with datatypes na tama na

        if len(df_latest) < MIN_REQUIRED_ROWS:
            print(f"⚠️ Node '{node}' has insufficient data ({len(df_latest)} rows). Skipping prediction.")
            job_fail(conn, node, ts_for_insert_and_queue, reason="Insufficient data for prediction")
            continue  # move to next job

        df_latest = enforce_fixed_interval(df_latest, "5min") #expanded and with proper resample values
        
        #print(df_latest)

        # Step 2: Get last raw timestamp (most recent sensor timestamp)
        last_raw_ts = ts_for_insert_and_queue

        # Step 3: Predict next temperature using lagged input
        predict_ts, predict_value = predict_next_step(
            model=initial_model,
            df_recent=df_latest,
            last_raw_ts=last_raw_ts,
            n_lags=3
        )

        if predict_value is not None:
            print(f"✅ Predicted {predict_value:.2f}°C for {predict_ts}")

            try:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO tbl_predicted_and_timestamp (node_name, predicted_timestamp, predicted_temperature)
                    VALUES (%s, %s, %s)
                """, (node, predict_ts, predict_value))
                conn.commit()
                cur.close()
                print("🗄 Prediction saved to database.")
            except Exception as e:
                print("⚠️ Failed to save prediction:", e)

            job_success(conn, node, ts_for_insert_and_queue)
            made = made + 1
            
            if made >= RETRAIN_AFTER:
                print("🔁 Retraining model with latest data...")
                cur = conn.cursor(dictionary=True)
                cur.execute("""
                    SELECT timestamp, temperature, humidity, node_1, node_2
                    FROM dht11_random_forest
                    ORDER BY timestamp DESC;
                """)
                rows = cur.fetchall()
                cur.close()

                df_all = pd.DataFrame(rows)
                df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
                df_all = df_all.set_index("timestamp")

                df_all = enforce_fixed_interval(df_all, "5min")
                df_all = make_lag_features(df_all, n_lags=3)
                df_all = take_training_window(df_all, 500)

                model = train_model(df_all)
                if model is not None:
                    print("✅ Model retrained successfully.")
                    made = 0
        else:
            print("❌ Prediction failed.")
            job_fail(conn, node, ts_for_insert_and_queue, reason="Prediction step failed")


# -----------------------------
# MAIN ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    try:
        conn = pool.get_connection()
        worker_loop(conn)
    except Exception as e:
        print("❌ Error:", e)
    finally:
        if conn and conn.is_connected():
            conn.close()
