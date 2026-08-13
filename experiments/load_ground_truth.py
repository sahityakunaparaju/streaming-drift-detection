import pandas as pd
import psycopg2


conn = psycopg2.connect(
    host="127.0.0.1",
    port=5432,
    dbname="drift_db",
    user="drift",
    password="drift",
)

df = pd.read_csv(
    "results/raw/ground_truth.csv"
)

cur = conn.cursor()

cur.execute(
    "TRUNCATE TABLE ground_truth CASCADE;"
)

for _, row in df.iterrows():

    cur.execute(
        """
        INSERT INTO ground_truth
        (
            event_id,
            ts,
            drift_type,
            subtype,
            feature,
            severity,
            injected,
            ground_truth
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            row["event_id"],
            float(row["timestamp"]),
            row["drift_type"],
            row["subtype"],
            row["feature"],
            row["severity"],
            bool(row["injected"]),
            row["ground_truth"],
        ),
    )

conn.commit()

cur.close()
conn.close()

print("Ground truth loaded successfully.")