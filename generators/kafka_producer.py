import json
import os
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer

STREAM_DIR = Path("results/raw/stream")
TOPIC = "iot-drift-stream"

BOOTSTRAP_SERVERS = os.environ.get(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092",
)

def load_semantic_metadata(window_file):
    """
    Load the semantic metadata sidecar produced by stream_builder.

    Returns None when the window has no semantic metadata.
    """

    metadata_path = (
        window_file.parent
        / f"{window_file.stem}.metadata.json"
    )

    if not metadata_path.exists():
        return None

    with open(metadata_path, "r") as file:
        return json.load(file)

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
)

window_files = sorted(
    STREAM_DIR.glob("window_*.csv")
)

print("=" * 60)
print("KAFKA PRODUCER")
print("=" * 60)

print(f"Topic: {TOPIC}")
print(f"Bootstrap: {BOOTSTRAP_SERVERS}")
print(f"Windows: {len(window_files)}")

for window_file in window_files:

    df = pd.read_csv(window_file)

    window_id = window_file.stem

    semantic_metadata = load_semantic_metadata(
        window_file
    )

    payload = {
        "event_id": window_id,
        "ts": float(df["ts"].iloc[0]),
        "records": df.to_dict(orient="records"),
        "semantic_metadata": semantic_metadata,
    }

    print(
        f"Sending {window_id} "
        f"({len(df)} records, "
        f"semantic_metadata="
        f"{semantic_metadata is not None})..."
    )

    future = producer.send(
        TOPIC,
        value=payload,
    )

    metadata = future.get(timeout=30)

    print(
        f"  partition={metadata.partition}, "
        f"offset={metadata.offset}"
    )

    time.sleep(0.5)

producer.flush()
producer.close()

print("=" * 60)
print("PRODUCER COMPLETED")
print("=" * 60)
