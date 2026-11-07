
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path.home() / "Documents" / "MapMatchingProject"
DATA_DIR = PROJECT_DIR / "data" / "raw" / "Geolife Trajectories 1.3" / "Data"
OUTPUT_DIR = PROJECT_DIR / "results" / "figures"

USER = "128"
MODE = "car"

def main():
    label_path = DATA_DIR / USER / "labels.txt"
    trajectory_dir = DATA_DIR / USER / "Trajectory"

    labels = pd.read_csv(label_path, sep="\t")
    labels.columns = [str(c).strip() for c in labels.columns]

    labels["Transportation Mode"] = (
        labels["Transportation Mode"].astype(str).str.strip().str.lower()
    )
    labels["start"] = pd.to_datetime(labels["Start Time"], errors="coerce")
    labels["end"] = pd.to_datetime(labels["End Time"], errors="coerce")

    labels = labels[
        (labels["Transportation Mode"] == MODE)
        & labels["start"].notna()
        & labels["end"].notna()
    ].copy()

    if labels.empty:
        print(f"No {MODE} labels found for user {USER}.")
        return

    # Start with the longest labelled interval.
    labels["duration_minutes"] = (
        labels["end"] - labels["start"]
    ).dt.total_seconds() / 60

    trip = labels.sort_values("duration_minutes", ascending=False).iloc[0]

    print(f"User: {USER}")
    print(f"Mode: {MODE}")
    print(f"Start: {trip['start']}")
    print(f"End: {trip['end']}")
    print(f"Label duration: {trip['duration_minutes']:.1f} minutes")

    points = []

    for path in sorted(trajectory_dir.glob("*.plt")):
        try:
            df = pd.read_csv(path, skiprows=6, header=None)

            if df.shape[1] < 7:
                continue

            df = df.iloc[:, :7].copy()
            df.columns = [
                "latitude", "longitude", "unused",
                "altitude", "days", "date", "time"
            ]

            df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
            df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
            df["timestamp"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str),
                errors="coerce"
            )

            df = df.dropna(
                subset=["latitude", "longitude", "timestamp"]
            )

            matched = df[
                (df["timestamp"] >= trip["start"])
                & (df["timestamp"] <= trip["end"])
            ].copy()

            if not matched.empty:
                points.append(matched)

        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")

    if not points:
        print("No GPS points found in this interval.")
        return

    gps = pd.concat(points, ignore_index=True)
    gps = gps.sort_values("timestamp").drop_duplicates(
        subset=["timestamp", "latitude", "longitude"]
    )

    if len(gps) < 2:
        print("Too few GPS points to plot a route.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = PROJECT_DIR / "data" / "processed" / "selected_trip.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    gps.to_csv(csv_path, index=False)

    plt.figure(figsize=(9, 8))
    plt.plot(
        gps["longitude"], gps["latitude"],
        linewidth=1, marker=".", markersize=3
    )
    plt.scatter(
        gps["longitude"].iloc[0], gps["latitude"].iloc[0],
        marker="o", s=80, label="Start"
    )
    plt.scatter(
        gps["longitude"].iloc[-1], gps["latitude"].iloc[-1],
        marker="X", s=80, label="End"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"GeoLife {MODE.title()} Trajectory — User {USER}")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    image_path = OUTPUT_DIR / "selected_trip.png"
    plt.savefig(image_path, dpi=200)
    plt.close()

    print(f"Matched GPS points: {len(gps)}")
    print(f"Saved GPS points to: {csv_path}")
    print(f"Saved plot to: {image_path}")

if __name__ == "__main__":
    main()
