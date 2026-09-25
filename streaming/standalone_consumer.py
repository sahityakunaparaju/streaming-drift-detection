"""
Standalone Kafka consumer for Phase 11 overhead measurement.

Runs inside the Docker network, connects to Kafka and Postgres
using the same detector code as the Spark consumer. Supports
two modes controlled by DETECTION_MODE environment variable:

  DETECTION_MODE=on   -- runs all detectors (schema, statistical, semantic, unified)
  DETECTION_MODE=off  -- passthrough, no detector computation

This avoids the Spark Structured Streaming Kafka connector JAR
compatibility issues while still measuring the actual detector
computation overhead on real batches through real Kafka -> Postgres.
"""

import json
import os
import sys
import time
import io

# Force unbuffered stdout/stderr so logs appear in real time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)

sys.path.insert(0, "/tmp")

import numpy as np
import pandas as pd
import psycopg2
from kafka import KafkaConsumer, TopicPartition

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KAFKA_BOOTSTRAP = "kafka:29092"
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "iot-drift-stream")
DETECTION_MODE = os.environ.get("DETECTION_MODE", "on")  # "on" or "off"

REFERENCE_CSV = "/tmp/iot_telemetry_data.csv"
PROFILE_PATH = "/tmp/reference_profile.json"

POSTGRES_HOST = "postgres"
POSTGRES_PORT = 5432
POSTGRES_DB = "drift_db"
POSTGRES_USER = "drift"
POSTGRES_PASSWORD = "drift"

# Timing log path (read by the host to measure detection overhead)
TIMING_LOG = "/tmp/detection_timing.jsonl"

# ---------------------------------------------------------------------------
# Load reference data
# ---------------------------------------------------------------------------

print(f"Loading reference data from {REFERENCE_CSV}...")
reference_df = pd.read_csv(REFERENCE_CSV)

with open(PROFILE_PATH, "r") as f:
    reference_profile = json.load(f)

reference_schema = reference_profile["schema"]
reference_contract = reference_profile["semantic_contract"]

print(f"Reference: {len(reference_df)} rows, "
      f"{len(reference_df.columns)} columns")
print(f"Detection mode: {DETECTION_MODE}")

# ---------------------------------------------------------------------------
# Detector imports (only loaded when mode=on)
# ---------------------------------------------------------------------------

if DETECTION_MODE == "on":
    from detectors.schema_detector import compare_schemas
    from detectors.statistical_detector import compare_numeric_fields
    from detectors.semantic_detector import (
        compare_semantic_contracts,
        validate_values_against_contract,
    )
    from detectors.unified import unified_score, decide
    from detectors.thresholds import THRESHOLDS

    CATEGORICAL_P_VALUE_THRESHOLD = float(
        THRESHOLDS["statistical"]["categorical_p_value_threshold"]
    )
    print("All detectors loaded (schema, statistical, semantic, unified)")


# ---------------------------------------------------------------------------
# Helper functions (mirrored from spark_consumer.py)
# ---------------------------------------------------------------------------

def build_current_schema(df):
    """Build a {dtype, nullable} schema reference from a live batch."""
    columns = {}
    for column in df.columns:
        dtype = str(df[column].dtype)
        if dtype.startswith("float"):
            dtype_name = "float64"
        elif dtype.startswith("int"):
            dtype_name = "int64"
        elif dtype == "bool":
            dtype_name = "bool"
        else:
            dtype_name = "string"
        columns[column] = {
            "dtype": dtype_name,
            "nullable": bool(df[column].isna().any()),
        }
    return {"schema_version": "stream", "columns": columns}


def categorical_drift(reference, current):
    """Chi-square drift test for one categorical column."""
    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()
    categories = sorted(
        set(ref_counts.index.astype(str))
        | set(cur_counts.index.astype(str))
    )
    if not categories:
        return {"drift": False, "chi2": 0.0, "p_value": 1.0}

    ref_values = [int(ref_counts.get(c, 0)) for c in categories]
    cur_values = [int(cur_counts.get(c, 0)) for c in categories]
    observed = np.array([ref_values, cur_values], dtype=float)

    if observed.sum() == 0:
        return {"drift": False, "chi2": 0.0, "p_value": 1.0}

    from scipy.stats import chi2_contingency
    chi2, p_value, _, _ = chi2_contingency(observed)
    return {
        "drift": bool(p_value < CATEGORICAL_P_VALUE_THRESHOLD),
        "chi2": float(chi2),
        "p_value": float(p_value),
    }


def make_semantic_contract(metadata):
    """Reference contract with the metadata sidecar's unit change applied."""
    contract = json.loads(json.dumps(reference_contract))
    if not metadata:
        return contract
    column = metadata.get("column")
    new_unit = metadata.get("new_unit")
    if column and new_unit and column in contract["fields"]:
        contract["fields"][column]["unit"] = new_unit
    return contract


def run_detectors(current_df, semantic_metadata):
    """Run the full detection pipeline. Returns (decision, timing_ms)."""
    t0 = time.perf_counter()

    # Schema detection
    current_schema = build_current_schema(current_df)
    schema_events = compare_schemas(reference_schema, current_schema)
    schema_drift = len(schema_events) > 0

    # Statistical detection
    statistical_results = {}
    numeric_columns = [
        col for col in reference_df.columns
        if col in current_df.columns
        and pd.api.types.is_numeric_dtype(reference_df[col])
        and pd.api.types.is_numeric_dtype(current_df[col])
        and col != "ts"
    ]
    if numeric_columns:
        try:
            statistical_results = compare_numeric_fields(
                reference_df, current_df, numeric_columns
            )
        except Exception as e:
            statistical_results = {"error": str(e)}

    # Categorical detection
    categorical_results = {}
    if "device" in reference_df.columns and "device" in current_df.columns:
        categorical_results["device"] = categorical_drift(
            reference_df["device"], current_df["device"]
        )

    # Semantic detection
    current_contract = make_semantic_contract(semantic_metadata)
    semantic_contract_events = compare_semantic_contracts(
        reference_contract, current_contract
    )
    semantic_value_events = validate_values_against_contract(
        current_df, current_contract
    )
    semantic_events = semantic_contract_events + semantic_value_events
    semantic_drift = len(semantic_events) > 0

    # Unified scoring
    statistical_details = {
        "numeric": statistical_results,
        "categorical": categorical_results,
    }
    scores = unified_score(schema_drift, statistical_details, semantic_drift)
    decision = decide(
        scores["unified_score"],
        schema_score=scores["schema_score"],
        semantic_score=scores["semantic_score"],
        statistical_score=scores["statistical_score"],
    )

    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000

    return {
        "decision": decision,
        "scores": scores,
        "schema_drift": schema_drift,
        "semantic_drift": semantic_drift,
        "statistical_results": statistical_results,
        "timing_ms": elapsed_ms,
    }


def store_in_postgres(cur, event_id, ts, detection_result):
    """Write one event to detection_events table."""
    temp_stats = detection_result.get("statistical_results", {})
    if not isinstance(temp_stats, dict):
        temp_stats = {}
    scores = detection_result["scores"]

    cur.execute(
        """
        INSERT INTO detection_events
        (event_id, batch_ts, schema_drift_flag, schema_details,
         schema_score, statistical_ks, statistical_psi,
         statistical_details, statistical_score,
         semantic_drift_flag, semantic_details, semantic_score,
         unified_score, decision)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            event_id,
            ts,
            detection_result["schema_drift"],
            json.dumps({"drift": detection_result["schema_drift"]}),
            scores["schema_score"],
            temp_stats.get("temp", {}).get("ks_statistic")
            if isinstance(temp_stats.get("temp"), dict) else None,
            temp_stats.get("temp", {}).get("psi")
            if isinstance(temp_stats.get("temp"), dict) else None,
            json.dumps({"numeric": temp_stats}),
            scores["statistical_score"],
            detection_result["semantic_drift"],
            json.dumps({"drift": detection_result["semantic_drift"]}),
            scores["semantic_score"],
            scores["unified_score"],
            detection_result["decision"],
        ),
    )


# ---------------------------------------------------------------------------
# Main consumer loop
# ---------------------------------------------------------------------------

def main():
    """Consume Kafka batches, run detection, persist events to Postgres."""
    print(f"\nConnecting to Kafka at {KAFKA_BOOTSTRAP}...")
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=60000,  # 60s idle timeout
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    print(f"Connected. Subscribed to '{KAFKA_TOPIC}'")
    print(f"Detection mode: {DETECTION_MODE}")
    print(f"Waiting for messages...\n")

    event_count = 0
    total_detection_ms = 0.0
    total_records = 0

    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )
    cur = conn.cursor()

    try:
        while True:
            # poll returns dict of {TopicPartition: [ConsumerRecord]}
            records = consumer.poll(timeout_ms=5000)

            if not records:
                # Signal idleness to the host via a sentinel file
                if event_count > 0:
                    with open("/tmp/consumer_idle.signal", "w") as f:
                        f.write(f"idle after {event_count} events\n")
                continue

            # Remove idle signal if it exists
            try:
                os.remove("/tmp/consumer_idle.signal")
            except FileNotFoundError:
                pass

            for tp, msgs in records.items():
                for msg in msgs:
                    payload = msg.value
                    event_id = payload["event_id"]
                    ts = payload["ts"]
                    records_data = payload["records"]
                    semantic_metadata = payload.get("semantic_metadata")

                    n_records = len(records_data) if records_data else 0
                    total_records += n_records

                    current_df = pd.DataFrame(records_data)

                    if DETECTION_MODE == "on":
                        result = run_detectors(current_df, semantic_metadata)
                        detection_ms = result["timing_ms"]
                    else:
                        # Passthrough mode: no detection
                        detection_ms = 0.0
                        result = {
                            "decision": "NO_DRIFT",
                            "scores": {
                                "unified_score": 0.0,
                                "schema_score": 0.0,
                                "statistical_score": 0.0,
                                "semantic_score": 0.0,
                            },
                            "schema_drift": False,
                            "semantic_drift": False,
                            "statistical_results": {},
                            "timing_ms": 0.0,
                        }

                    store_in_postgres(cur, event_id, ts, result)
                    conn.commit()

                    event_count += 1
                    total_detection_ms += detection_ms

                    # Write timing record to JSONL for host-side measurement
                    receive_ts = time.time()
                    timing_record = {
                        "event_id": event_id,
                        "send_ts": ts,
                        "receive_ts": receive_ts,
                        "n_records": n_records,
                        "detection_ms": detection_ms,
                        "mode": DETECTION_MODE,
                    }
                    with open(TIMING_LOG, "a") as tf:
                        tf.write(json.dumps(timing_record) + "\n")

                    if event_count % 5 == 0 or event_count <= 3:
                        print(
                            f"  [{event_count}] {event_id}: "
                            f"{n_records} records, "
                            f"detection={detection_ms:.1f}ms, "
                            f"decision={result['decision']}"
                        )

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        consumer.close()
        cur.close()
        conn.close()

    print(f"\n{'='*60}")
    print(f"CONSUMER SUMMARY")
    print(f"{'='*60}")
    print(f"Mode: {DETECTION_MODE}")
    print(f"Events processed: {event_count}")
    print(f"Total records: {total_records}")
    print(f"Total detection time: {total_detection_ms:.1f} ms")
    if event_count > 0:
        print(f"Mean detection time per event: "
              f"{total_detection_ms / event_count:.1f} ms")
        if total_records > 0:
            print(f"Mean detection time per 1000 rows: "
                  f"{total_detection_ms / total_records * 1000:.1f} ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
