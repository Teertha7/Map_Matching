
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path.home() / "Documents" / "MapMatchingProject"

GEOLIFE_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "Geolife Trajectories 1.3"
    / "Data"
)

def main():
    print("=" * 60)
    print("GEOLIFE DATASET AUDIT")
    print("=" * 60)

    if not GEOLIFE_DIR.exists():
        print("\nERROR: Dataset folder not found.")
        print("Expected location:")
        print(GEOLIFE_DIR)
        print("\nCheck the folder name and location.")
        return

    user_dirs = sorted(
        p for p in GEOLIFE_DIR.iterdir()
        if p.is_dir() and p.name.isdigit()
    )

    print(f"\nDataset path: {GEOLIFE_DIR}")
    print(f"Number of user folders: {len(user_dirs)}")

    total_files = 0
    total_points = 0
    valid_files = 0
    user_summaries = []

    for user_dir in user_dirs:
        trajectory_dir = user_dir / "Trajectory"

        if not trajectory_dir.exists():
            continue

        plt_files = sorted(trajectory_dir.glob("*.plt"))
        user_points = 0
        user_valid_files = 0

        for file in plt_files:
            total_files += 1

            try:
                # GeoLife .plt files contain 6 header lines.
                df = pd.read_csv(
                    file,
                    skiprows=6,
                    header=None
                )

                if df.shape[1] < 7:
                    continue

                df = df.iloc[:, :7]
                df.columns = [
                    "latitude",
                    "longitude",
                    "unused",
                    "altitude",
                    "days",
                    "date",
                    "time"
                ]

                df["latitude"] = pd.to_numeric(
                    df["latitude"], errors="coerce"
                )
                df["longitude"] = pd.to_numeric(
                    df["longitude"], errors="coerce"
                )

                df = df.dropna(
                    subset=["latitude", "longitude"]
                )

                # Basic coordinate sanity check.
                df = df[
                    df["latitude"].between(-90, 90)
                    & df["longitude"].between(-180, 180)
                ]

                point_count = len(df)

                if point_count > 0:
                    user_points += point_count
                    total_points += point_count
                    valid_files += 1
                    user_valid_files += 1

            except Exception as error:
                print(f"Could not read {file.name}: {error}")

        user_summaries.append({
            "user": user_dir.name,
            "trajectory_files": len(plt_files),
            "valid_files": user_valid_files,
            "gps_points": user_points
        })

    summary = pd.DataFrame(user_summaries)

    print(f"\nTotal trajectory files: {total_files}")
    print(f"Files with valid GPS points: {valid_files}")
    print(f"Total valid GPS points: {total_points:,}")

    if not summary.empty:
        print("\nTop 10 users by number of GPS points:")
        print(
            summary.sort_values(
                "gps_points", ascending=False
            ).head(10).to_string(index=False)
        )

        output_file = PROJECT_DIR / "results" / "tables"
        output_file.mkdir(parents=True, exist_ok=True)

        summary.to_csv(
            output_file / "dataset_audit.csv",
            index=False
        )

        print(
            "\nSaved user-level summary to:",
            output_file / "dataset_audit.csv"
        )

    print("\nAudit complete.")

if __name__ == "__main__":
    main()
