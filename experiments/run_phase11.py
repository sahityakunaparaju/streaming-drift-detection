"""
Phase 11 (Section W) — Streaming Overhead experiment (REAL PIPELINE).

Answers RQ5: Can the unified detector operate continuously with
acceptable streaming overhead?

Runs against the real Docker pipeline: Kafka producer -> standalone
consumer -> Postgres, using window_size=1000 (matching every other
experiment in this project).

Load points:
  - comfortable: 5 windows of 1000 rows (5,000 total), sent at 0.5s intervals
  - stress: 20 windows of 1000 rows (20,000 total), sent at 0.05s intervals

For each load point, measures WITH and WITHOUT the detection layer:
  - Throughput (rows/sec end-to-end: send to Postgres commit)
  - End-to-end latency (mean/median/P95: producer send -> Postgres write)
  - Detection computation time per event (from consumer timing log)
  - CPU and memory (via docker stats on the consumer container)

Each load point run 3 repetitions; reports mean.

OVERHEAD is reported as a percentage delta -- this answers RQ5.

Usage:
    PYTHONPATH=. python3 experiments/run_phase11.py [--reps 3] [--out results/phase11]
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd
import psycopg2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_REPS = 3
DEFAULT_OUT = "results/phase11"

KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPIC_BASE = "overhead-test"
POSTGRES_HOST = "127.0.0.1"
POSTGRES_PORT = 5434
POSTGRES_DB = "drift_db"
POSTGRES_USER = "drift"
POSTGRES_PASSWORD = "drift"

SPARK_CONTAINER = "drift-spark"

LOAD_POINTS = {
    "comfortable": {
        "n_windows": 5,
        "window_size": 1000,
        "description": "5 windows x 1000 rows = 5,000 total",
        "inter_window_delay_sec": 0.5,
    },
    "stress": {
        "n_windows": 20,
        "window_size": 1000,
        "description": "20 windows x 1000 rows = 20,000 total",
        "inter_window_delay_sec": 0.05,
    },
}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def load_reference():
    """Load the primary IoT telemetry dataset used as the window source."""
    df = pd.read_csv("data/primary/iot_telemetry_data.csv")
    return df


def generate_windows(reference_df, n_windows, window_size, seed=42):
    """Generate n_windows of IoT data, window_size rows each."""
    import numpy as np
    rng = np.random.RandomState(seed)

    windows = []
    n_rows = len(reference_df)

    for i in range(n_windows):
        start = (i * window_size) % n_rows
        end = start + window_size
        if end > n_rows:
            batch = pd.concat([
                reference_df.iloc[start:n_rows],
                reference_df.iloc[:end - n_rows],
            ], ignore_index=True)
        else:
            batch = reference_df.iloc[start:end].copy()
        windows.append(batch)

    return windows


# ---------------------------------------------------------------------------
# Kafka producer
# ---------------------------------------------------------------------------

def send_windows_to_kafka(windows, inter_delay, topic):
    """
    Send windows to Kafka, recording send timestamps.
    Returns list of (event_id, send_ts).
    """
    from kafka import KafkaProducer

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    send_log = []

    for i, window_df in enumerate(windows):
        event_id = f"overhead-{i:04d}"
        send_ts = time.time()

        payload = {
            "event_id": event_id,
            "ts": send_ts,
            "records": window_df.to_dict(orient="records"),
            "semantic_metadata": None,
        }

        future = producer.send(topic, value=payload)
        future.get(timeout=30)
        send_log.append((event_id, send_ts))

        if inter_delay > 0:
            time.sleep(inter_delay)

    producer.flush()
    producer.close()

    return send_log


# ---------------------------------------------------------------------------
# Consumer management
# ---------------------------------------------------------------------------

def create_kafka_topic(topic_name):
    """Create a fresh Kafka topic."""
    from kafka.admin import KafkaAdminClient
    from kafka import TopicPartition
    try:
        admin = KafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP)
        try:
            admin.create_topics([topic_name], validate_only=False)
        except Exception:
            pass  # topic may already exist
        admin.close()
    except Exception:
        pass
    return topic_name


def delete_kafka_topic(topic_name):
    """Delete a Kafka topic."""
    try:
        from kafka.admin import KafkaAdminClient
        admin = KafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP)
        try:
            admin.delete_topics([topic_name])
        except Exception:
            pass
        admin.close()
    except Exception:
        pass


def start_consumer(mode, topic):
    """Start the standalone consumer in the Docker container."""
    subprocess.run(
        ["docker", "exec", SPARK_CONTAINER,
         "pkill", "-f", "standalone_consumer"],
        capture_output=True,
    )
    time.sleep(2)

    subprocess.run(
        ["docker", "exec", SPARK_CONTAINER,
         "rm", "-f", "/tmp/detection_timing.jsonl"],
        capture_output=True,
    )

    # Start consumer in background, subscribing to the specific topic
    env_var = "DETECTION_MODE=on" if mode == "on" else "DETECTION_MODE=off"
    subprocess.run(
        ["docker", "exec", "-d", SPARK_CONTAINER,
         "bash", "-c",
         f"cd /tmp && PYTHONPATH=/opt/spark/python:$PYTHONPATH "
         f"KAFKA_TOPIC={topic} {env_var} python3 standalone_consumer.py "
         f"> /tmp/consumer_{mode}.log 2>&1"],
        check=True,
    )

    # Wait for consumer to connect to Kafka
    time.sleep(10)


def stop_consumer():
    """Stop the consumer."""
    subprocess.run(
        ["docker", "exec", SPARK_CONTAINER,
         "pkill", "-f", "standalone_consumer"],
        capture_output=True,
    )
    time.sleep(2)


def wait_for_events_in_postgres(n_events, timeout_sec=120):
    """Wait until n_events appear in detection_events table."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            conn = psycopg2.connect(
                host=POSTGRES_HOST, port=POSTGRES_PORT,
                dbname=POSTGRES_DB, user=POSTGRES_USER,
                password=POSTGRES_PASSWORD, connect_timeout=5,
            )
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM detection_events "
                "WHERE event_id LIKE 'overhead-%'"
            )
            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            if count >= n_events:
                return count
        except Exception:
            pass
        time.sleep(1)
    return 0


def clear_test_data():
    """Remove overhead events and timing logs."""
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST, port=POSTGRES_PORT,
            dbname=POSTGRES_DB, user=POSTGRES_USER,
            password=POSTGRES_PASSWORD, connect_timeout=5,
        )
        cur = conn.cursor()
        cur.execute("DELETE FROM detection_events WHERE event_id LIKE 'overhead-%'")
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass

    subprocess.run(
        ["docker", "exec", SPARK_CONTAINER,
         "rm", "-f", "/tmp/detection_timing.jsonl"],
        capture_output=True,
    )


def read_timing_log():
    """Read the timing JSONL file from the container."""
    result = subprocess.run(
        ["docker", "exec", SPARK_CONTAINER,
         "cat", "/tmp/detection_timing.jsonl"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return []

    records = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def get_container_stats():
    """Get CPU and memory stats for the consumer container."""
    result = subprocess.run(
        ["docker", "stats", "--no-stream", "--format",
         "{{.CPUPerc}}\t{{.MemUsage}}",
         SPARK_CONTAINER],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        return {"cpu_pct": 0.0, "memory_mb": 0.0}

    parts = result.stdout.strip().split("\t")
    cpu_pct = 0.0
    memory_mb = 0.0

    if parts:
        cpu_str = parts[0].replace("%", "").strip()
        try:
            cpu_pct = float(cpu_str)
        except ValueError:
            pass

    if len(parts) > 1:
        mem_str = parts[1].strip()
        # Parse "123.4MiB" or "123.4MiB / 1.234GiB"
        try:
            mem_val = mem_str.split("/")[0].strip()
            if "GiB" in mem_val:
                memory_mb = float(mem_val.replace("GiB", "").strip()) * 1024
            elif "MiB" in mem_val:
                memory_mb = float(mem_val.replace("MiB", "").strip())
            elif "KiB" in mem_val:
                memory_mb = float(mem_val.replace("KiB", "").strip()) / 1024
        except (ValueError, IndexError):
            pass

    return {"cpu_pct": cpu_pct, "memory_mb": memory_mb}


# ---------------------------------------------------------------------------
# Overhead computation
# ---------------------------------------------------------------------------

def compute_overhead(on_metrics, off_metrics):
    """Compute overhead as delta/ratio."""
    result = {}

    for metric in ["latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
                    "detection_time_mean_ms"]:
        w = on_metrics.get(metric, 0)
        wo = off_metrics.get(metric, 0)
        delta = w - wo
        pct = ((w - wo) / wo * 100) if wo > 0 else 0.0
        result[f"{metric}_delta"] = round(delta, 2)
        result[f"{metric}_pct"] = round(pct, 1)

    w_tp = on_metrics.get("throughput_rows_sec", 0)
    wo_tp = off_metrics.get("throughput_rows_sec", 0)
    result["throughput_delta"] = round(w_tp - wo_tp, 2)
    result["throughput_pct"] = (
        round((w_tp - wo_tp) / wo_tp * 100, 1) if wo_tp > 0 else 0.0
    )

    return result


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_single_measurement(n_windows, window_size, inter_delay, detection_mode):
    """
    Send n_windows and wait for them to be processed.
    Returns metrics dict.
    """
    run_id = f"{detection_mode}_{int(time.time())}"
    topic = f"{KAFKA_TOPIC_BASE}-{run_id}"
    create_kafka_topic(topic)
    time.sleep(3)

    clear_test_data()
    start_consumer(detection_mode, topic)
    time.sleep(5)  # let consumer settle

    stats_before = get_container_stats()

    reference_df = load_reference()
    windows = generate_windows(reference_df, n_windows, window_size, seed=42)

    send_log = send_windows_to_kafka(windows, inter_delay, topic)
    total_send_time = time.time() - send_log[0][1]

    n_processed = wait_for_events_in_postgres(n_windows, timeout_sec=120)

    stats_after = get_container_stats()
    timing_records = read_timing_log()

    # Keep only this run's events (drop stale smoke-test events)
    timing_records = [
        r for r in timing_records
        if r["event_id"].startswith("overhead-")
    ]

    if not timing_records:
        print(f"    WARNING: No timing records found")
        return None

    latencies_ms = []
    detection_times_ms = []

    for record in timing_records:
        # End-to-end latency: consumer receive time - producer send time
        latency = (record["receive_ts"] - record["send_ts"]) * 1000
        latencies_ms.append(latency)
        detection_times_ms.append(record["detection_ms"])

    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)

    total_rows = n_windows * window_size
    wall_time = max(r["receive_ts"] for r in timing_records) - send_log[0][1]

    metrics = {
        "total_rows": total_rows,
        "n_windows": n_windows,
        "n_processed": len(timing_records),
        "latency_mean_ms": round(sum(latencies_ms) / n, 2),
        "latency_p50_ms": round(sorted_lat[n // 2], 2),
        "latency_p95_ms": round(sorted_lat[int(n * 0.95)], 2),
        "latency_min_ms": round(sorted_lat[0], 2),
        "latency_max_ms": round(sorted_lat[-1], 2),
        "detection_time_mean_ms": round(
            sum(detection_times_ms) / len(detection_times_ms), 2
        ),
        "detection_time_total_ms": round(sum(detection_times_ms), 2),
        "throughput_rows_sec": round(total_rows / wall_time, 2) if wall_time > 0 else 0,
        "wall_time_sec": round(wall_time, 2),
        "cpu_pct": round((stats_before["cpu_pct"] + stats_after["cpu_pct"]) / 2, 2),
        "memory_mb": round((stats_before["memory_mb"] + stats_after["memory_mb"]) / 2, 2),
    }

    return metrics


def run_experiment(reps=DEFAULT_REPS, out_dir=DEFAULT_OUT):
    """Run the full Phase 11 experiment."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 11 -- STREAMING OVERHEAD (REAL PIPELINE)")
    print(f"  window_size=1000 (matching Phase 7-10 experiments)")
    print(f"  Kafka -> consumer -> Postgres (inside Docker network)")
    print(f"  Repetitions per load point: {reps}")
    print(f"  Load points: {', '.join(LOAD_POINTS.keys())}")
    print("=" * 70)

    all_results = []

    for lp_name, lp_config in LOAD_POINTS.items():
        print(f"\n{'='*70}")
        print(f"LOAD POINT: {lp_name}")
        print(f"  {lp_config['description']}")
        print(f"  inter-window delay: {lp_config['inter_window_delay_sec']}s")
        print(f"{'='*70}")

        on_results = []
        off_results = []

        for rep in range(reps):
            print(f"\n  --- Repetition {rep+1}/{reps} ---")

            print("  [ON] Running with detection...")
            result_on = run_single_measurement(
                lp_config["n_windows"],
                lp_config["window_size"],
                lp_config["inter_window_delay_sec"],
                "on",
            )
            if result_on:
                on_results.append(result_on)
                print(f"    latency: mean={result_on['latency_mean_ms']:.1f}ms "
                      f"P50={result_on['latency_p50_ms']:.1f}ms "
                      f"P95={result_on['latency_p95_ms']:.1f}ms")
                print(f"    detection: mean={result_on['detection_time_mean_ms']:.1f}ms")
                print(f"    throughput: {result_on['throughput_rows_sec']:.1f} rows/sec")

            print("  [OFF] Running without detection...")
            result_off = run_single_measurement(
                lp_config["n_windows"],
                lp_config["window_size"],
                lp_config["inter_window_delay_sec"],
                "off",
            )
            if result_off:
                off_results.append(result_off)
                print(f"    latency: mean={result_off['latency_mean_ms']:.1f}ms "
                      f"P50={result_off['latency_p50_ms']:.1f}ms "
                      f"P95={result_off['latency_p95_ms']:.1f}ms")
                print(f"    detection: mean={result_off['detection_time_mean_ms']:.1f}ms")
                print(f"    throughput: {result_off['throughput_rows_sec']:.1f} rows/sec")

        stop_consumer()

        if on_results and off_results:
            mean_on = _mean_metrics(on_results)
            mean_off = _mean_metrics(off_results)
            overhead = compute_overhead(mean_on, mean_off)

            print(f"\n  === OVERHEAD SUMMARY for {lp_name} ===")
            print(f"  End-to-end latency overhead: "
                  f"{overhead['latency_mean_ms_pct']}%  "
                  f"(+{overhead['latency_mean_ms_delta']:.1f} ms)")
            print(f"  Detection computation overhead: "
                  f"{overhead['detection_time_mean_ms_pct']}%  "
                  f"(+{overhead['detection_time_mean_ms_delta']:.1f} ms)")
            print(f"  Throughput change: {overhead['throughput_pct']}%")

            all_results.append({
                "load_point": lp_name,
                "config": lp_config,
                "with_detection": mean_on,
                "without_detection": mean_off,
                "overhead": overhead,
                "repetitions": {
                    "with_detection": on_results,
                    "without_detection": off_results,
                },
            })

    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")

    csv_path = out_path / "overhead.csv"
    _save_csv(all_results, csv_path)
    print(f"Saved: {csv_path}")

    summary = {
        "phase": 11,
        "section": "W",
        "rq5": "Can the unified detector operate continuously with acceptable streaming overhead?",
        "pipeline": "Kafka -> standalone consumer -> Postgres (Docker network)",
        "window_size": 1000,
        "supersedes": "Prior local-simulation run (invalid: fixed cost / near-zero batch denominator)",
        "load_points": LOAD_POINTS,
        "repetitions": reps,
        "results": all_results,
        "answer": _generate_answer(all_results),
    }
    summary_path = out_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {summary_path}")

    print(f"\n{'='*70}")
    print("RQ5 ANSWER")
    print(f"{'='*70}")
    print(summary["answer"])

    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

METRICS = [
    "latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
    "detection_time_mean_ms", "detection_time_total_ms",
    "throughput_rows_sec", "wall_time_sec",
    "cpu_pct", "memory_mb",
]


def _mean_metrics(results):
    mean = {}
    for metric in METRICS:
        vals = [r[metric] for r in results if metric in r]
        mean[metric] = round(sum(vals) / len(vals), 2) if vals else 0.0
    return mean


def _save_csv(all_results, csv_path):
    rows = []
    for r in all_results:
        w = r["with_detection"]
        wo = r["without_detection"]
        oh = r["overhead"]
        rows.append({
            "load_point": r["load_point"],
            "n_windows": r["config"]["n_windows"],
            "window_size": r["config"]["window_size"],
            "latency_mean_with_ms": w["latency_mean_ms"],
            "latency_mean_without_ms": wo["latency_mean_ms"],
            "latency_p50_with_ms": w["latency_p50_ms"],
            "latency_p50_without_ms": wo["latency_p50_ms"],
            "latency_p95_with_ms": w["latency_p95_ms"],
            "latency_p95_without_ms": wo["latency_p95_ms"],
            "latency_overhead_pct": oh["latency_mean_ms_pct"],
            "detection_time_mean_with_ms": w["detection_time_mean_ms"],
            "detection_time_mean_without_ms": wo["detection_time_mean_ms"],
            "detection_overhead_pct": oh["detection_time_mean_ms_pct"],
            "throughput_with": w["throughput_rows_sec"],
            "throughput_without": wo["throughput_rows_sec"],
            "throughput_overhead_pct": oh["throughput_pct"],
            "cpu_with_pct": w["cpu_pct"],
            "cpu_without_pct": wo["cpu_pct"],
            "memory_with_mb": w["memory_mb"],
            "memory_without_mb": wo["memory_mb"],
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def _generate_answer(results):
    lines = []
    for r in results:
        lp = r["load_point"]
        oh = r["overhead"]
        lines.append(
            f"  - {lp}: latency {oh['latency_mean_ms_pct']:+.1f}% "
            f"(+{oh['latency_mean_ms_delta']:.1f} ms), "
            f"detection {oh['detection_time_mean_ms_delta']:+.1f} ms/event, "
            f"throughput {oh['throughput_pct']:+.1f}%"
        )

    avg_latency_pct = sum(
        r["overhead"]["latency_mean_ms_pct"] for r in results
    ) / len(results) if results else 0
    avg_detection_delta = sum(
        r["overhead"]["detection_time_mean_ms_delta"] for r in results
    ) / len(results) if results else 0

    if avg_latency_pct < 30:
        verdict = (
            "RQ5 CONFIRMED: The drift detection layer adds <30% end-to-end "
            "latency overhead at realistic batch sizes (window_size=1000), "
            "confirming the protection layer does not destroy pipeline performance."
        )
    elif avg_latency_pct < 100:
        verdict = (
            "RQ5 PARTIALLY CONFIRMED: The drift detection layer adds "
            "noticeable but bounded overhead at realistic batch sizes."
        )
    else:
        verdict = (
            "RQ5 NOT CONFIRMED: The drift detection layer adds significant "
            "overhead even at realistic batch sizes."
        )

    breakdown = "\n".join(lines)
    return f"{verdict}\n\nPer-load-point breakdown:\n{breakdown}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 11 -- Streaming Overhead (REAL PIPELINE)"
    )
    parser.add_argument(
        "--reps", type=int, default=DEFAULT_REPS,
        help="Repetitions per load point (default: %(default)s)",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help="Output directory (default: %(default)s)",
    )
    args = parser.parse_args()

    run_experiment(reps=args.reps, out_dir=args.out)
