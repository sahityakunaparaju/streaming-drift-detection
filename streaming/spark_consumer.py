import sys
sys.path.insert(0, "/tmp")

import json
import psycopg2
import pandas as pd
import numpy as np

from scipy.stats import chi2_contingency

from pyspark.sql import SparkSession

from detectors.schema_detector import compare_schemas
from detectors.statistical_detector import compare_numeric_fields
from detectors.semantic_detector import (
    compare_semantic_contracts,
    validate_values_against_contract,
)
from detectors.unified import (
    unified_score,
    decide,
)
from detectors.thresholds import THRESHOLDS

CATEGORICAL_P_VALUE_THRESHOLD = float(
    THRESHOLDS["statistical"]["categorical_p_value_threshold"]
)

KAFKA_BOOTSTRAP = "kafka:29092"
KAFKA_TOPIC = "iot-drift-stream"

REFERENCE_CSV = "/tmp/iot_telemetry_data.csv"
PROFILE_PATH = "/tmp/reference_profile.json"

reference_df = pd.read_csv(REFERENCE_CSV)

with open(PROFILE_PATH, "r") as f:
    reference_profile = json.load(f)

reference_schema = reference_profile["schema"]
reference_contract = reference_profile["semantic_contract"]

spark = (
    SparkSession.builder
    .appName("IoTDriftConsumer")
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

    return {
        "schema_version": "stream",
        "columns": columns,
    }

def categorical_drift(reference, current):
    """
    Chi-square comparison of categorical distributions.
    Returns a simple drift result.
    """

    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()

    categories = sorted(
        set(ref_counts.index.astype(str))
        | set(cur_counts.index.astype(str))
    )

    if not categories:
        return {
            "drift": False,
            "chi2": 0.0,
            "p_value": 1.0,
        }

    ref_values = []
    cur_values = []

    for category in categories:
        ref_values.append(
            int(ref_counts.get(category, 0))
        )
        cur_values.append(
            int(cur_counts.get(category, 0))
        )

    observed = np.array(
        [ref_values, cur_values],
        dtype=float,
    )

    if observed.sum() == 0:
        return {
            "drift": False,
            "chi2": 0.0,
            "p_value": 1.0,
        }

    chi2, p_value, _, _ = chi2_contingency(observed)

    return {
        "drift": bool(
            p_value < CATEGORICAL_P_VALUE_THRESHOLD
        ),
        "chi2": float(chi2),
        "p_value": float(p_value),
    }

def make_semantic_contract(metadata):
    """
    Start from the reference contract and apply
    the semantic metadata carried by Kafka.
    """

    contract = json.loads(
        json.dumps(reference_contract)
    )

    if not metadata:
        return contract

    column = metadata.get("column")
    new_unit = metadata.get("new_unit")

    if (
        column
        and new_unit
        and column in contract["fields"]
    ):
        contract["fields"][column]["unit"] = new_unit

    return contract

def process_batch(batch_df, batch_id):
    """Run detection on one Spark micro-batch and persist the event."""
    rows = (
        batch_df
        .selectExpr("CAST(value AS STRING) AS value")
        .collect()
    )

    print()
    print("=" * 60)
    print(f"Processing Spark batch {batch_id}")
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
        semantic_metadata = payload.get(
            "semantic_metadata"
        )

        print(
            f"\nReceived {event_id}: "
            f"{len(records)} records"
        )

        print(
            f"records type: {type(records)}"
        )

        if records:
            print(
                f"first record type: "
                f"{type(records[0])}"
            )

        current_df = pd.DataFrame(records)

        current_schema = build_current_schema(
            current_df
        )

        schema_events = compare_schemas(
            reference_schema,
            current_schema,
        )

        schema_drift = len(schema_events) > 0

        schema_result = {
            "drift": schema_drift,
            "events": schema_events,
        }

        statistical_results = {}

        numeric_columns = [
            column
            for column in reference_df.columns
            if column in current_df.columns
            and pd.api.types.is_numeric_dtype(
                reference_df[column]
            )
            and pd.api.types.is_numeric_dtype(
                current_df[column]
            )
            and column != "ts"
        ]

        if numeric_columns:

            try:
                statistical_results = (
                    compare_numeric_fields(
                        reference_df,
                        current_df,
                        numeric_columns,
                    )
                )

            except Exception as e:

                print(
                    "  [ERROR] Statistical detector "
                    f"failed for {event_id}: {e}"
                )

                statistical_results = {
                    "error": str(e)
                }

        categorical_results = {}

        if (
            "device" in reference_df.columns
            and "device" in current_df.columns
        ):

            categorical_results["device"] = (
                categorical_drift(
                    reference_df["device"],
                    current_df["device"],
                )
            )

        current_contract = make_semantic_contract(
            semantic_metadata
        )

        semantic_contract_events = (
            compare_semantic_contracts(
                reference_contract,
                current_contract,
            )
        )

        semantic_value_events = (
            validate_values_against_contract(
                current_df,
                current_contract,
            )
        )

        semantic_events = (
            semantic_contract_events
            + semantic_value_events
        )

        semantic_drift = (
            len(semantic_events) > 0
        )

        semantic_result = {
            "drift": semantic_drift,
            "contract_events": (
                semantic_contract_events
            ),
            "value_events": (
                semantic_value_events
            ),
        }

        statistical_details = {
            "numeric": statistical_results,
            "categorical": categorical_results,
        }

        scores = unified_score(
            schema_drift,
            statistical_details,
            semantic_drift,
        )

        decision = decide(
            scores["unified_score"],
            schema_score=scores["schema_score"],
            semantic_score=scores["semantic_score"],
            statistical_score=scores["statistical_score"],
        )

        print(
            f"  Schema drift: "
            f"{schema_drift}"
        )

        print(
            f"  Semantic drift: "
            f"{semantic_drift}"
        )

        print(
            f"  Statistical columns: "
            f"{list(statistical_results.keys())}"
        )

        print(
            f"  Unified score: "
            f"{scores['unified_score']:.3f}"
        )

        print(
            f"  Decision: {decision}"
        )

        temp_stats = (
            statistical_results.get("temp", {})
            if isinstance(
                statistical_results,
                dict
            )
            else {}
        )

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
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )
            """,
            (
                event_id,
                ts,

                schema_drift,
                json.dumps(schema_result),
                scores["schema_score"],

                temp_stats.get(
                    "ks_statistic"
                ),
                temp_stats.get(
                    "psi"
                ),

                json.dumps(
                    statistical_details
                ),
                scores["statistical_score"],

                semantic_drift,
                json.dumps(
                    semantic_result
                ),
                scores["semantic_score"],

                scores["unified_score"],
                decision,
            ),
        )

    conn.commit()

    cur.close()
    conn.close()

    print()
    print(
        f"Stored {len(rows)} event(s) "
        "in PostgreSQL"
    )

query = (
    df_stream
    .writeStream
    .foreachBatch(process_batch)
    .option(
        "checkpointLocation",
        "/tmp/drift-consumer-checkpoint",
    )
    .start()
)

query.awaitTermination()
