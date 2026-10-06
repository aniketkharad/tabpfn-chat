"""Generate synthetic industrial equipment sensor anomaly dataset (120 rows, ~6.7% failure rate)."""

from pathlib import Path
import numpy as np
import pandas as pd

def generate():
    np.random.seed(42)
    n_total = 120
    n_anomalies = 8  # ~6.7% rare anomalies
    n_normal = n_total - n_anomalies

    # Normal equipment operating distribution
    normal_temp = np.random.normal(65.0, 3.5, n_normal).clip(55, 75)
    normal_vib = np.random.normal(2.2, 0.4, n_normal).clip(1.2, 3.2)
    normal_press = np.random.normal(5.0, 0.3, n_normal).clip(4.2, 5.8)
    normal_rpm = np.random.normal(1800, 40, n_normal).clip(1700, 1900)
    normal_hours = np.random.uniform(200, 4500, n_normal)
    normal_failure = np.zeros(n_normal, dtype=int)

    # Anomalous operating regime (overheating, severe vibration, pressure spikes)
    anom_temp = np.random.normal(92.0, 5.0, n_anomalies).clip(82, 105)
    anom_vib = np.random.normal(6.5, 0.8, n_anomalies).clip(5.0, 8.5)
    anom_press = np.random.normal(8.2, 0.6, n_anomalies).clip(7.0, 9.5)
    anom_rpm = np.random.normal(2150, 60, n_anomalies).clip(2000, 2300)
    anom_hours = np.random.uniform(3500, 6000, n_anomalies)
    anom_failure = np.ones(n_anomalies, dtype=int)

    temp = np.concatenate([normal_temp, anom_temp])
    vib = np.concatenate([normal_vib, anom_vib])
    press = np.concatenate([normal_press, anom_press])
    rpm = np.concatenate([normal_rpm, anom_rpm])
    hours = np.concatenate([normal_hours, anom_hours])
    failure = np.concatenate([normal_failure, anom_failure])

    # Shuffle deterministically
    indices = np.random.permutation(n_total)
    
    sensor_ids = [f"SNS_{i+1:03d}" for i in range(n_total)]

    df = pd.DataFrame({
        "sensor_id": sensor_ids,
        "temperature_c": np.round(temp[indices], 1),
        "vibration_mms": np.round(vib[indices], 2),
        "pressure_bar": np.round(press[indices], 2),
        "rpm": np.round(rpm[indices]).astype(int),
        "operating_hours": np.round(hours[indices]).astype(int),
        "is_failure": failure[indices],
    })

    out_dir = Path("demos/04_anomaly_detection")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "equipment_anomalies.csv"
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} rows ({df['is_failure'].sum()} failures) -> {csv_path}")

    # Also save to static samples
    static_samples = Path("src/tabchat/frontend/static/samples")
    static_samples.mkdir(parents=True, exist_ok=True)
    df.to_csv(static_samples / "equipment_anomalies.csv", index=False)

if __name__ == "__main__":
    generate()
