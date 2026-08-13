from drift_injection.stream_builder import build_stream


build_stream(
    input_path="data/primary/iot_telemetry_data.csv",
    output_dir="results/raw/stream",
    ground_truth_path="results/raw/ground_truth.csv",
    window_size=1000,
    max_windows=10,
)

print("Stream generation completed.")