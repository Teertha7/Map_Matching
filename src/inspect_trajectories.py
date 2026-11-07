
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path.home() / "Documents" / "MapMatchingProject"

GEOLIFE_DIR = (
    PROJECT_DIR / "data" / "raw"
    / "Geolife Trajectories 1.3" / "Data"
)

def load_trajectory(file_path):
    df = pd.read_csv(file_path, skiprows=6, header=None)

    if df.shape[1] < 7:
        return None

    df = df.iloc[:, :7].copy()
    df.columns = [
        "latitude", "longitude", "unused",
        "altitude", "days", "date", "time"
    ]

    df["latitude"] = pd.to_numeric(
        df["latitude"], errors="coerce"
    )
    df["longitude"] = pd.to_numeric(
        df["longitude"], errors="coerce"
    )

    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce"
    )

    df = df.dropna(
        subset=["latitude", "longitude", "timestamp"]
    )

    df = df[
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    ].copy()

    return df.sort_values("timestamp").reset_index(drop=True)


def main():
    files = sorted(GEOLIFE_DIR.glob("*/Trajectory/*.plt"))

    if not files:
        print("No trajectory files found. Check GEOLIFE_DIR.")
        return

    records = []

    # Inspect up to 100 files initially.
    for file_path in files[:100]:
        df = load_trajectory(file_path)

        if df is None or len(df) < 2:
            continue

        elapsed = (
            df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
        ).total_seconds()

        # Approximate distance between consecutive GPS points.
        lat = np.radians(df["latitude"].to_numpy())
        lon = np.radians(df["longitude"].to_numpy())

        dlat = np.diff(lat)
        dlon = np.diff(lon)

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat[:-1]) * np.cos(lat[1:])
            * np.sin(dlon / 2) ** 2
        )

        distances_m = (
            6371000 * 2
            * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        )

        time_diffs = (
            df["timestamp"].diff().dt.total_seconds().to_numpy()[1:]
        )

        valid = time_diffs > 0
        speeds_kmh = (
            distances_m[valid] / time_diffs[valid] * 3.6
        )

        records.append({
            "user": file_path.parts[-3],
            "file": file_path.name,
            "points": len(df),
            "duration_hours": elapsed / 3600,
            "median_interval_seconds": (
                np.median(time_diffs[valid]) if valid.any()
                else np.nan
            ),
            "median_speed_kmh": (
                np.median(speeds_kmh) if len(speeds_kmh)
                else np.nan
            ),
            "latitude_min": df["latitude"].min(),
            "latitude_max": df["latitude"].max(),
            "longitude_min": df["longitude"].min(),
            "longitude_max": df["longitude"].max()
        })

    summary = pd.DataFrame(records)

    if summary.empty:
        print("No usable trajectories in the sample.")
        return

    output_dir = PROJECT_DIR / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "trajectory_sample.csv"
    summary.to_csv(output_path, index=False)

    print("\nSample trajectory summary:")
    print(summary.describe().round(2).to_string())

    print("\nFirst 10 trajectories:")
    print(summary.head(10).to_string(index=False))

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
