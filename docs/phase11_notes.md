# Phase 11 — Streaming Overhead (Section W)

## RQ5

**Can the unified detector operate continuously with acceptable streaming overhead?**

The claim to prove is narrow and modest: "the protection layer doesn't destroy pipeline performance" — NOT that this system outperforms raw Spark throughput.

---

## Supersedes Prior Run

The prior Phase 11 run used a **local simulation** with unrealistic batch sizes (1-6 rows per batch), producing meaningless overhead percentages (+40,260% latency) driven purely by dividing a fixed ~1.6s detector cost by a near-zero batch size. This was invalid and is superseded by the real-pipeline measurements below.

---

## Pipeline

- **Kafka broker** (Apache Kafka 4.0.0 in Docker)
- **Consumer** (Python kafka-python, running inside Docker network, same detector code as the Spark consumer)
- **Postgres** (PostgreSQL 16 in Docker)

The consumer runs the same detector pipeline as `streaming/spark_consumer.py`: schema comparison, statistical tests (KS/PSI), semantic contract validation, and unified scoring. The only difference from the Spark consumer is the scheduling mechanism (direct poll vs. Spark micro-batch).

---

## Load Points

| Load Point | Windows | Window Size | Total Rows | Inter-Window Delay | Justification |
|------------|---------|-------------|------------|--------------------|---------------|
| **Comfortable** | 5 | 1,000 | 5,000 | 0.5s | Normal sustained operation; 2 windows/sec arrival rate vs. detector capacity |
| **Stress** | 20 | 1,000 | 20,000 | 0.05s | Burst traffic; 20 windows/sec arrival rate, exceeding detector capacity |

**window_size=1000** matches every other experiment in this project (Phases 7-10).

---

## Results

### Per-Load-Point Summary

| Metric | Comfortable | Stress |
|--------|-------------|--------|
| **Detection ON latency (mean)** | **1,123 ms** | **4,376 ms** |
| **Detection OFF latency (mean)** | **671 ms** | **438 ms** |
| **Latency overhead** | **+67%** (+452 ms) | **+898%** (+3,938 ms) |
| Detection time per window | 540 ms | 507 ms |
| Detection ON throughput | 1,209 rows/sec | 1,527 rows/sec |
| Detection OFF throughput | 1,387 rows/sec | 3,556 rows/sec |
| Throughput overhead | -12.8% | -57.1% |

### Per-Repetition Detail (Comfortable)

| Rep | ON mean (ms) | OFF mean (ms) | Overhead | Detection (ms) |
|-----|-------------|---------------|----------|----------------|
| 1 | 1,226 | 697 | +76% | 575 |
| 2 | 1,355 | 639 | +112% | 589 |
| 3 | 789 | 677 | +17% | 455 |
| **Mean** | **1,123** | **671** | **+67%** | **540** |

### Per-Repetition Detail (Stress)

| Rep | ON mean (ms) | OFF mean (ms) | Overhead | Detection (ms) |
|-----|-------------|---------------|----------|----------------|
| 1 | 4,566 | 677 | +575% | 507 |
| 2 | 5,009 | 274 | +1,727% | 509 |
| 3 | 3,553 | 365 | +874% | 503 |
| **Mean** | **4,376** | **438** | **+898%** | **507** |

---

## Mechanism Analysis

### Why comfortable overhead is +67%

The detector takes **~540 ms per 1,000-row window** (schema comparison + KS/PSI statistical tests against the 405K-row reference + semantic contract validation + unified scoring). At 2 windows/second arrival rate, detection can keep up but each window adds its detection time to its latency:

```
Detection ON latency  ≈ baseline (Kafka + Postgres) + detection computation
                      ≈ 671ms + 540ms ≈ 1,211ms  (observed: 1,123ms)
```

The +67% overhead is the detector's fixed per-window cost at a sustainable arrival rate. This is **bounded and predictable**.

### Why stress overhead is +898% (queuing)

At 20 windows/second arrival rate but only ~2 windows/second detection capacity (1 / 500ms), a **backlog** forms:

```
Window  1: arrives t=0.00, detection completes t=0.54, latency = 540ms
Window  2: arrives t=0.05, detection starts t=0.54, completes t=1.08, latency = 1,030ms
Window  3: arrives t=0.10, detection starts t=1.08, completes t=1.62, latency = 1,520ms
...
Window 20: arrives t=0.95, detection starts t=9.45, completes t=9.99, latency = 9,040ms
```

The last window's latency is the sum of all queuing + its own detection time. This is a **queuing effect**, not a detector efficiency problem — any sequential processor with fixed per-item cost will exhibit this behavior under excess load.

### CPU and Memory

| Metric | Detection ON | Detection OFF |
|--------|-------------|---------------|
| CPU (container) | 1.2% | 1.0% |
| Memory (container) | 545 MB | 519 MB |

CPU and memory overhead from detection is negligible (~0.2% CPU, ~26 MB memory). The bottleneck is **detection computation time**, not resource utilization.

---

## RQ5 Answer

> **At comfortable load (5 windows of 1,000 rows, 0.5s apart):**
> Detection adds +67% end-to-end latency (+452 ms) and -12.8% throughput.
> The detector's fixed cost of ~540 ms per window is bounded and predictable.
>
> **At stress load (20 windows of 1,000 rows, 0.05s apart):**
> Detection adds +898% end-to-end latency due to queuing — windows arrive
> faster than the detector can process them (20/sec arrival vs. 2/sec capacity).
>
> **RQ5: PARTIALLY CONFIRMED.** At sustained operational rates (≤2 windows/sec),
> the detection layer adds ~67% latency — noticeable but bounded. At burst rates
> exceeding the detector's throughput capacity (~2 windows/sec), queuing causes
> unbounded latency growth. The detector does not "destroy pipeline performance"
> at normal operating rates, but it does introduce a throughput ceiling of
> ~2,000 rows/sec (1,000 rows × 2 windows/sec). Above this rate, a buffering
> or sampling strategy would be needed.
>
> **Key number for the paper:** Detection adds ~540 ms per 1,000-row window,
> equivalent to ~0.54 ms per row of pure detector computation overhead.

---

## Method Notes

- **Out of scope**: No detector logic, thresholds, or accuracy-related code was modified.
- **Repetitions**: 3 per load point.
- **Mode**: Real pipeline inside Docker network (Kafka → consumer → Postgres).
- **Consumer**: Uses kafka-python (not Spark Structured Streaming) due to Spark 4.1.2 Kafka connector JAR compatibility issues. The detector code is identical; only the scheduling mechanism differs.
- **Sanity check**: The +898% stress overhead was investigated and traced to a clear queuing mechanism (arrival rate > service rate), not a measurement artifact.

---

## Test Suite Status

| Metric | Value |
|--------|-------|
| Tests passing before Phase 11 | 129 |
| Tests passing after Phase 11 | 129 |
| New tests added | 0 (measurement phase, no new detector logic) |

---

## Files

| File | Description |
|------|-------------|
| `experiments/run_phase11.py` | Experiment runner |
| `streaming/standalone_consumer.py` | Docker consumer (ON/OFF modes, timing logs) |
| `streaming/spark_consumer_passthrough.py` | Passthrough consumer (unused, kept for reference) |
| `results/phase11/overhead.csv` | Machine-readable results |
| `results/phase11/summary.json` | Full experiment summary |
| `docs/phase11_notes.md` | This document |
