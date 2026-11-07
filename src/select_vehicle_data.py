
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path.home() / "Documents" / "MapMatchingProject"
DATA_DIR = PROJECT_DIR / "data" / "raw" / "Geolife Trajectories 1.3" / "Data"
OUTPUT_DIR = PROJECT_DIR / "results" / "tables"

def main():
    records = []

    for label_file in sorted(DATA_DIR.glob("*/labels.txt")):
        user = label_file.parent.name

        try:
            labels = pd.read_csv(label_file, sep="\t")
        except Exception as exc:
            print(f"Skipping {label_file}: {exc}")
            continue

        labels.columns = [str(c).strip() for c in labels.columns]

        required = {
            "Start Time", "End Time", "Transportation Mode"
        }
        if not required.issubset(labels.columns):
            print(f"Unexpected columns for user {user}:")
            print(labels.columns.tolist())
            continue

        labels["Transportation Mode"] = (
            labels["Transportation Mode"].astype(str).str.strip().str.lower()
        )

        vehicle_labels = labels[
            labels["Transportation Mode"].isin(["car", "taxi"])
        ].copy()

        for _, row in vehicle_labels.iterrows():
            start = pd.to_datetime(
                row["Start Time"], errors="coerce"
            )
            end = pd.to_datetime(
                row["End Time"], errors="coerce"
            )

            if pd.isna(start) or pd.isna(end) or end < start:
                continue

            records.append({
                "user": user,
                "mode": row["Transportation Mode"],
                "start_time": start,
                "end_time": end,
                "duration_minutes": (end - start).total_seconds() / 60
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not records:
        print("No valid car/taxi label intervals found.")
        return

    segments = pd.DataFrame(records)
    segments.to_csv(OUTPUT_DIR / "vehicle_label_segments.csv", index=False)

    print(f"Valid car/taxi-labelled segments: {len(segments)}")
    print("\nSegments by mode:")
    print(segments["mode"].value_counts().to_string())

    print("\nVehicle-labelled segments by user:")
    user_summary = (
        segments.groupby("user")
        .agg(
            segments=("mode", "size"),
            car_segments=("mode", lambda x: (x == "car").sum()),
            taxi_segments=("mode", lambda x: (x == "taxi").sum()),
            total_labelled_minutes=("duration_minutes", "sum")
        )
        .sort_values("total_labelled_minutes", ascending=False)
    )

    print(user_summary.head(15).round(1).to_string())

    print("\nDate range:")
    print("Start:", segments["start_time"].min())
    print("End:  ", segments["end_time"].max())

    print("\nSaved segment inventory to:")
    print(OUTPUT_DIR / "vehicle_label_segments.csv")

if __name__ == "__main__":
    main()
