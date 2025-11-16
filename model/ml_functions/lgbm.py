import pandas as pd
import lightgbm as lgb
from helpers.db_connection import pool
import configparser

config = configparser.ConfigParser()
config.read('server/setting.conf')

# FETCHING DATA
def fetch_rows_upto(node_name, ts, limit=250):
    conn = pool.get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT timestamp, temperature, humidity, node_name, site_name
            FROM vw_node_site_readings
            WHERE node_name=%s AND timestamp <= %s
            ORDER BY timestamp DESC
            LIMIT %s
        """, (node_name, ts, limit))
        rows = cur.fetchall()
        rows = list(reversed(rows))
        return rows
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()

# QUEUE MANAGEMENT
def claim_job(conn):
    cur = conn.cursor(dictionary=True)
    cur.execute("START TRANSACTION")
    cur.execute("""
        SELECT node_name, ts FROM tbl_queue
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
        UPDATE tbl_queue
        SET status='processing', attempt=attempt+1
        WHERE node_name=%s AND ts=%s
    """, (row["node_name"], row["ts"]))
    cur.execute("COMMIT")
    cur.close()
    return row


def job_success(conn, node_name, ts):
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE tbl_queue
            SET status='done', completed_at=NOW()
            WHERE node_name=%s AND ts=%s
        """, (node_name, ts))
        conn.commit()
        cur.close()
        print(f"Job success recorded for node '{node_name}' at {ts}")
    except Exception as e:
        print(f"Failed to mark job success for {node_name}: {e}")


def job_fail(conn, node_name, ts, reason=None):
    try:
        cur = conn.cursor()
        if reason:
            cur.execute("""
                UPDATE tbl_queue
                SET status='failed', completed_at=NOW(), fail_reason=%s
                WHERE node_name=%s AND ts=%s
            """, (reason, node_name, ts))
        else:
            cur.execute("""
                UPDATE tbl_queue
                SET status='failed', completed_at=NOW()
                WHERE node_name=%s AND ts=%s
            """, (node_name, ts))
        conn.commit()
        cur.close()
        print(f"Job failed for node '{node_name}' at {ts}. Reason: {reason or 'Unknown'}")
    except Exception as e:
        print(f"Failed to mark job as failed for {node_name}: {e}")

# CLEANING / RESAMPLING
def clean_dataframe(rows):
    if rows is None or len(rows) == 0:
        empty_df = pd.DataFrame(columns=["temperature", "humidity", "node_name", "site_name"])
        empty_df.index = pd.DatetimeIndex([], name="timestamp")
        return empty_df

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")

    for c in ["node_name", "site_name"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    df = df.dropna(subset=["temperature", "humidity"])
    return df


def enforce_fixed_interval(df, frequency):
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")] # removes duplicates pero iwan yung last

    df = df.resample(frequency).agg({
        "temperature": "mean",
        "humidity": "mean",
        "node_name": "last",
        "site_name": "last",
    })

    df[["temperature", "humidity"]] = df[["temperature", "humidity"]].interpolate(method="time")
    df = df.ffill().bfill() #fill forware and backward
    return df

# FEATURES
def make_lag_features(df, n_lags):
    feat = df.copy()
    for lag_number in range(1, n_lags + 1):
        feat[f"temp_lag{lag_number}"] = feat["temperature"].shift(lag_number) #+1 or higher will move/shift data downward 
        feat[f"hum_lag{lag_number}"] = feat["humidity"].shift(lag_number)
    feat["target_next_temp"] = feat["temperature"].shift(-1) # -1 moves/shift data upward
    need_cols = []

    for c in feat.columns:
        if c.startswith("temp_lag") or c.startswith("hum_lag"):
            need_cols.append(c)
    need_cols.append("target_next_temp")
    need_cols.append("node_name")
    need_cols.append("site_name")
    feat = feat.dropna(subset=need_cols)
    return feat

# TRAINING (LightGBM)
def train_model(df):
    df = df.copy()
    min_rows = config.getint('lgbm_model', 'MIN_REQUIRED_ROWS')
    if len(df) < min_rows:
        print(f"Not enough data to train. Need {min_rows}, got {len(df)}.")
        return None, None, None

    print("\nDEBUG: Training DataFrame head()")
    print(df.head(5))
    print("Columns:", df.columns.tolist(), "\n")

    lag_cols = []                          # start with an empty list

    for c in df.columns:                   # loop through every column name in df
        if c.startswith("temp_lag") or c.startswith("hum_lag"):   # check the condition
            lag_cols.append(c)             # if it matches, add it to the list
    cat_cols = ["node_name", "site_name"]
    
    # Ensure categories are fixed
    categories_map = {}
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
            df[col] = df[col].cat.set_categories(df[col].unique())
            categories_map[col] = df[col].cat.categories

    X = df[lag_cols + cat_cols]
    y = df["target_next_temp"]

    train_data = lgb.Dataset(X, label=y, categorical_feature=cat_cols)

    params = {
        "objective": "regression", # numeric output
        "metric": "rmse",   # metric evalution during training error correction. just like auto arima's aic and bic scoring to get pdq's
        "boosting_type": "gbdt", #Gradient Boosted Decision Trees, the default and most widely used boosting method in LightGBM. An algo
        "num_leaves": 31, #More leaves = more complex model (higher accuracy but higher overfitting risk)
        "learning_rate": 0.05, #Smaller values slower but more accurate learning (safer) and Larger values faster but risk overshooting or overfitting.
        "feature_fraction": 0.9, #uses 90 of all available features. Other features is. mag cycle to sa lahat ng table randomly to use for bagging sample
        "bagging_fraction": 0.8, #use 80 of all total rows
        "bagging_freq": 5, #how many random 
        "verbose": -1
    }

    model = lgb.train(params, train_data, num_boost_round=100)#num_boost_round is the number of iteration for correcting the last prediction of lgbm to build the final prediction
    return model, categories_map, list(X.columns)

# PREDICTION
def prepare_predict_input(df, n_lags, categories_map=None, feature_columns=None): #builds the row input
    feature_dict = {}
    for lag_number in range(1, n_lags + 1):
        feature_dict[f"temp_lag{lag_number}"] = float(df["temperature"].iloc[-lag_number])
        feature_dict[f"hum_lag{lag_number}"] = float(df["humidity"].iloc[-lag_number])
    
    last = df.iloc[-1]
    feature_dict["node_name"] = last["node_name"]
    feature_dict["site_name"] = last["site_name"]

    X = pd.DataFrame([feature_dict])

    if categories_map:
        for col, cats in categories_map.items():
            X[col] = X[col].astype("category")
            X[col] = X[col].cat.set_categories(cats)

    if feature_columns:
        X = X[feature_columns]  # ensure exact order

    return X



def predict_next_step(model_bundle, df_recent, last_raw_ts, n_lags=3):
    model, categories_map, feature_columns = model_bundle
    if model is None:
        return None, None

    X = prepare_predict_input(df_recent, n_lags, categories_map, feature_columns)
    if X is None:
        return None, None

    y_pred = model.predict(X)[0]
    next_ts = last_raw_ts + pd.to_timedelta(config['lgbm_model']['FREQUENCY']) #add the frequency of prediction to the last ts to produce the predictions ts
    return next_ts, float(y_pred)

# TRAINING WINDOW (Cap Data)
def take_training_window(df, window_size):
    if len(df) > window_size:
        start_index = len(df) - window_size
        end_index = len(df)
        df_window = df.iloc[start_index:end_index]
        return df_window
    else:
        return df
    
#Insert Site ID on prediction table
def get_site_id_for_node(conn, node_name, ts_for_insert_and_queue):
    try:
        cur = conn.cursor()
        # If you have a nodes table:
        # cur.execute("SELECT site_id FROM tbl_nodes WHERE node_name=%s LIMIT 1", (node_name,))
        # Or derive from your view's latest row:
        cur.execute("""
            SELECT site_id
            FROM tbl_queue
            WHERE node_name=%s and ts = %s
            LIMIT 1
        """, (node_name,ts_for_insert_and_queue,))
        row = cur.fetchone()
        cur.close()
        return row[0] if row else None
    except Exception:
        try:
            cur.close()
        except Exception:
            pass
        return None

