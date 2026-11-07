
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path.home() / "Documents" / "MapMatchingProject"
DATA_DIR = PROJECT_DIR / "data" / "raw" / "Geolife Trajectories 1.3" / "Data"

def main():
    label_files = sorted(DATA_DIR.glob("*/labels.txt"))

    print(f"Users with labels.txt: {len(label_files)}")

    if not label_files:
        print("No label files found. Check the dataset structure.")
        return

    all_labels = []
    labelled_users = []

    for path in label_files:
        try:
            df = pd.read_csv(path, sep="\t")

            # Expected columns:
            # Start Time, End Time, Transportation Mode
            df.columns = [str(c).strip() for c in df.columns]

            if "Transportation Mode" not in df.columns:
                print(f"Unexpected format: {path}")
                print("Columns:", df.columns.tolist())
                continue

            df["user"] = path.parent.name
            all_labels.append(df)
            labelled_users.append(path.parent.name)

        except Exception as exc:
            print(f"Could not read {path}: {exc}")

    if not all_labels:
        print("No usable transportation labels found.")
        return

    labels = pd.concat(all_labels, ignore_index=True)
    labels["Transportation Mode"] = (
        labels["Transportation Mode"].astype(str).str.strip()
    )

    print(f"\nUsers with readable labels: {len(set(labelled_users))}")
    print(f"Total labelled segments: {len(labels)}")

    print("\nSegments by transportation mode:")
    print(labels["Transportation Mode"].value_counts().to_string())

    print("\nUsers with labels:")
    print(", ".join(sorted(set(labelled_users))))

    out_dir = PROJECT_DIR / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels.to_csv(out_dir / "transport_labels.csv", index=False)
    print(f"\nSaved labels to: {out_dir / 'transport_labels.csv'}")

if __name__ == "__main__":
    main()
