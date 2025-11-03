import time
import pandas as pd
from helpers.db_connection import pool
from model.ml_functions import rf
SLEEP_IDLE = 0.3
RETRAIN_AFTER = 10


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

    df_with_lags = rf.enforce_fixed_interval(df_all, rf.FREQUENCY)
    df_with_lags = rf.make_lag_features(df_with_lags, n_lags=3)

    df_with_clean = rf.clean_dataframe(df_with_lags) # ayusin yung datatypes

    df_with_cap = rf.take_training_window(df_with_clean, 500) #caps to 500

    print("Model: RANDOM FOREST: Train sets - ", len(df_with_cap))

    initial_model = rf.train_model(df_with_cap)

    if initial_model is not None:
        print("[Model trained successfully!]")
    else:
        print("[Model training failed!]")

    while True:
        job = rf.claim_job(conn)
        if not job:
            print("ℹ️ No job found, sleeping...")
            time.sleep(idle)
            idle = min(2.0, idle * 1.5)
            continue

        idle = SLEEP_IDLE
        node = job["node_name"]
        ts_for_insert_and_queue = pd.to_datetime(job["ts"])
        print(f"---Processing job from node '{node}' at {ts_for_insert_and_queue}")

         # Step 1: Fetch latest raw data again before predicting
        latest_rows = rf.fetch_latest_rows(node)

        df_latest = rf.clean_dataframe(latest_rows) #with datatypes na tama na

        if len(df_latest) < rf.MIN_REQUIRED_ROWS:
            print(f"***Node '{node}' has insufficient data ({len(df_latest)} rows). Skipping prediction.")
            rf.job_fail(conn, node, ts_for_insert_and_queue, reason="Insufficient data for prediction")
            continue  # move to next job

        df_latest = rf.enforce_fixed_interval(df_latest, rf.FREQUENCY) #expanded and with proper resample values
        
        #print(df_latest)

        # Step 2: Get last raw timestamp (most recent sensor timestamp)
        last_raw_ts = ts_for_insert_and_queue

        # Step 3: Predict next temperature using lagged input
        predict_ts, predict_value = rf.predict_next_step(
            model=initial_model,
            df_recent=df_latest,
            last_raw_ts=last_raw_ts,
            n_lags=3
        )

        if predict_value is not None:
            print(f"---Predicted {predict_value:.2f}°C for {predict_ts}")

            try:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO tbl_predicted_and_timestamp (node_name, predicted_timestamp, predicted_temperature)
                    VALUES (%s, %s, %s)
                """, (node, predict_ts, predict_value))
                conn.commit()
                cur.close()
                print("[Prediction saved to database!]")
            except Exception as e:
                print("Failed to save prediction:", e)

            rf.job_success(conn, node, ts_for_insert_and_queue)
            made = made + 1
            
            if made >= RETRAIN_AFTER:
                print("[Retraining model with latest data...]")
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

                df_all = rf.enforce_fixed_interval(df_all, rf.FREQUENCY)
                df_all = rf.make_lag_features(df_all, n_lags=3)
                df_all = rf.take_training_window(df_all, 500)

                model = rf.train_model(df_all)
                if model is not None:
                    print("[Model retrained successfully!]")
                    made = 0
        else:
            print("[Prediction failed!]")
            rf.job_fail(conn, node, ts_for_insert_and_queue, reason="Prediction step failed")


# -----------------------------
# MAIN ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    try:
        conn = pool.get_connection()
        worker_loop(conn)
    except Exception as e:
        print("Error:", e)
    finally:
        if conn and conn.is_connected():
            conn.close()
