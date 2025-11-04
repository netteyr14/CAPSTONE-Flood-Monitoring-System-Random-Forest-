import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from helpers.db_connection import pool
import configparser

config = configparser.ConfigParser()
config.read('server/setting.conf')

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
    if len(df) < config.getint('rf_model', 'MIN_REQUIRED_ROWS'):
        print(f"⚠️ Not enough data to train. Need {config.getint('rf_model', 'MIN_REQUIRED_ROWS')}, got {len(df)}.")
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
    next_timestamp = last_raw_ts + pd.to_timedelta(config.getint('rf_model', 'FREQUENCY'))
    # print(f"🧩 Predicted next temp = {y_pred:.2f}°C at {next_timestamp}")
    return next_timestamp, float(y_pred)