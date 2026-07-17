import os
from datetime import datetime, timedelta

import pandas as pd # type: ignore
import psycopg2 # type: ignore


def get_database_url():
    return os.environ.get("DATABASE_URL")


def get_connection(database_url: str = None):
    database_url = database_url or get_database_url()
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL not set. Set it as an environment variable (or "
            "Streamlit secret / GitHub Actions secret) pointing at your "
            "Supabase Postgres connection string."
        )
    return psycopg2.connect(database_url)


def _resolve_conn(conn, database_url):
    """Returns (connection, should_close). If conn was passed in, reuse it
    and don't close it here. Otherwise open a fresh one that we do close."""
    if conn is not None:
        return conn, False
    return get_connection(database_url), True


def init_log_table(database_url: str = None, conn=None):
    c, should_close = _resolve_conn(conn, database_url)
    cur = c.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS live_predictions_log (
            id SERIAL PRIMARY KEY,
            prediction_time TEXT NOT NULL,
            city TEXT NOT NULL,
            current_datetime TEXT NOT NULL,
            current_pm25 REAL,
            target_time_1h TEXT,
            target_time_24h TEXT,
            target_time_48h TEXT,
            pred_1h REAL,
            pred_24h REAL,
            pred_48h REAL,
            lower_1h REAL,
            upper_1h REAL,
            lower_24h REAL,
            upper_24h REAL,
            lower_48h REAL,
            upper_48h REAL,
            actual_1h REAL,
            actual_24h REAL,
            actual_48h REAL,
            error_1h REAL,
            error_24h REAL,
            error_48h REAL,
            UNIQUE (city, current_datetime)
        )
    """)
    c.commit()
    if should_close:
        c.close()


def log_prediction(city, current_dt, current_pm25, pred_1h, pred_24h, pred_48h,
                    lower_1h=None, upper_1h=None, lower_24h=None, upper_24h=None,
                    lower_48h=None, upper_48h=None, database_url: str = None, conn=None):
    c, should_close = _resolve_conn(conn, database_url)
    cur = c.cursor()

    prediction_time = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
    current_datetime_str = current_dt.strftime("%Y-%m-%d %H:%M:%S")
    target_1h = (current_dt + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    target_24h = (current_dt + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
    target_48h = (current_dt + timedelta(hours=48)).strftime("%Y-%m-%d %H:%M:%S")

    cur.execute("""
        INSERT INTO live_predictions_log
        (prediction_time, city, current_datetime, current_pm25,
         target_time_1h, target_time_24h, target_time_48h,
         pred_1h, pred_24h, pred_48h,
         lower_1h, upper_1h, lower_24h, upper_24h, lower_48h, upper_48h)
        VALUES (%s,%s,%s,%s, %s,%s,%s, %s,%s,%s, %s,%s,%s,%s,%s,%s)
        ON CONFLICT (city, current_datetime) DO NOTHING
    """, (
        prediction_time, city, current_datetime_str,
        round(current_pm25, 2) if current_pm25 is not None else None,
        target_1h, target_24h, target_48h,
        round(pred_1h, 2) if pred_1h else None,
        round(pred_24h, 2) if pred_24h else None,
        round(pred_48h, 2) if pred_48h else None,
        lower_1h, upper_1h, lower_24h, upper_24h, lower_48h, upper_48h,
    ))
    c.commit()
    if should_close:
        c.close()


def evaluate_past_predictions(live_df_by_city: dict, database_url: str = None, conn=None):
    """
    Only pulls rows that still have at least one unresolved horizon —
    (actual_1h IS NULL AND pred_1h IS NOT NULL) OR same for 24h/48h — instead
    of the entire table. Keeps this cheap regardless of how large the log
    has grown.
    """
    c, should_close = _resolve_conn(conn, database_url)
    query = """
        SELECT * FROM live_predictions_log
        WHERE (actual_1h IS NULL AND pred_1h IS NOT NULL)
           OR (actual_24h IS NULL AND pred_24h IS NOT NULL)
           OR (actual_48h IS NULL AND pred_48h IS NOT NULL)
    """
    log_df = pd.read_sql_query(query, c)
    now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    cur = c.cursor()
    changed = False

    for _, row in log_df.iterrows():
        city = row["city"]
        if city not in live_df_by_city:
            continue
        city_data = live_df_by_city[city].copy()
        city_data["datetime"] = pd.to_datetime(city_data["datetime"])

        for horizon in ["1h", "24h", "48h"]:
            if pd.notna(row.get(f"actual_{horizon}")) or pd.isna(row.get(f"pred_{horizon}")):
                continue
            target_time = pd.to_datetime(row[f"target_time_{horizon}"])
            if target_time > now_ist:
                continue
            city_data["time_diff"] = (city_data["datetime"] - target_time).abs()
            closest = city_data.nsmallest(1, "time_diff")
            if len(closest) > 0 and closest.iloc[0]["time_diff"] < timedelta(minutes=5):
                actual = round(float(closest.iloc[0]["pm2_5"]), 2)
                error = round(float(row[f"pred_{horizon}"]) - actual, 2)
                cur.execute(
                    f"UPDATE live_predictions_log SET actual_{horizon} = %s, error_{horizon} = %s WHERE id = %s",
                    (actual, error, int(row["id"])),
                )
                changed = True

    if changed:
        c.commit()
    if should_close:
        c.close()


def load_live_log(database_url: str = None, conn=None) -> pd.DataFrame:
    c, should_close = _resolve_conn(conn, database_url)
    df = pd.read_sql_query("SELECT * FROM live_predictions_log ORDER BY prediction_time", c)
    if should_close:
        c.close()
    if not df.empty:
        df["prediction_time"] = pd.to_datetime(df["prediction_time"])
        df["current_datetime"] = pd.to_datetime(df["current_datetime"])
    return df