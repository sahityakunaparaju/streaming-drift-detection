"""
Spark consumer WITHOUT drift detection -- pass-through baseline.

This consumer reads from Kafka, writes the raw event metadata
to Postgres, but does NOT run any detectors.  Used as the
"without detection" baseline for Phase 11 overhead measurements.
"""

import sys
sys.path.insert(0, "/tmp")

import json
import time
import psycopg2

from pyspark.sql import SparkSession

KAFKA_BOOTSTRAP = "kafka:29092"
KAFKA_TOPIC = "iot-drift-stream"

spark = (
    SparkSession.builder
    .appName("IoTDriftConsumer-Passthrough")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

df_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "earliest")
    .load()
)


def process_batch(batch_df, batch_id):
    """
    Pass-through: just write event metadata to Postgres.
    No detector computation at all.
    """

    rows = (
        batch_df
        .selectExpr("CAST(value AS STRING) AS value")
        .collect()
    )

    print()
    print("=" * 60)
    print(f"[PASS-THROUGH] Processing batch {batch_id}")
    print(f"Messages received: {len(rows)}")
    print("=" * 60)

    if not rows:
        return

    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        dbname="drift_db",
        user="drift",
        password="drift",
    )

    cur = conn.cursor()

    for row in rows:
        payload = json.loads(row["value"])

        event_id = payload["event_id"]
        ts = payload["ts"]
        records = payload["records"]
        n_records = len(records) if records else 0

        print(
            f"  Passthrough {event_id}: "
            f"{n_records} records"
        )

        # Store minimal metadata -- no detection results
        cur.execute(
            """
            INSERT INTO detection_events
            (
                event_id,
                batch_ts,
                schema_drift_flag,
                schema_details,
                schema_score,
                statistical_ks,
                statistical_psi,
                statistical_details,
                statistical_score,
                semantic_drift_flag,
                semantic_details,
                semantic_score,
                unified_score,
                decision
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """,
            (
                event_id,
                ts,
                False,              # schema_drift_flag
                json.dumps({}),     # schema_details
                0.0,                # schema_score
                None,               # statistical_ks
                None,               # statistical_psi
                json.dumps({}),     # statistical_details
                0.0,                # statistical_score
                False,              # semantic_drift_flag
                json.dumps({}),     # semantic_details
                0.0,                # semantic_score
                0.0,                # unified_score
                "NO_DRIFT",         # decision
            ),
        )

    conn.commit()
    cur.close()
    conn.close()

    print(
        f"  Stored {len(rows)} event(s) in PostgreSQL (passthrough)"
    )


query = (
    df_stream
    .writeStream
    .foreachBatch(process_batch)
    .option(
        "checkpointLocation",
        "/tmp/drift-consumer-passthrough-checkpoint",
    )
    .start()
)

query.awaitTermination()
