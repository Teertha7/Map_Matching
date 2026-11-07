#!/usr/bin/env python3
"""Extract valid GeoLife .plt trajectories into clean CSV files.

Run from the project root:
    python src/prepare_trips.py

Expected input:
    data/raw/Geolife Trajectories 1.3/Data/<user_id>/Trajectory/*.plt

Output:
    data/processed/trips/trip_001.csv, ...
    data/processed/trips/trip_summary.csv
"""
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "Geolife Trajectories 1.3" / "Data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "trips"
SUMMARY_PATH = OUTPUT_DIR / "trip_summary.csv"

MAX_TRAJECTORIES = 30
MIN_POINTS = 100


def read_geolife_plt(path: Path) -> pd.DataFrame:
    """Read the standard Geolife .plt format without mixing date/time columns."""
    # Standard .plt columns:
    # 0 latitude, 1 longitude, 2 unused, 3 altitude (feet),
    # 4 Excel-style date number, 5 date (YYYY-MM-DD), 6 time (HH:MM:SS).
    raw = pd.read_csv(
        path,
        skiprows=6,
        header=None,
        usecols=[0, 1, 3, 5, 6],
        dtype={0: "float64", 1: "float64", 3: "float64", 5: "string", 6: "string"},
        low_memory=False,
    )
    raw.columns = ["latitude", "longitude", "altitude_ft", "date", "time"]

    raw["timestamp"] = pd.to_datetime(
        raw["date"].str.strip() + " " + raw["time"].str.strip(),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    raw["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
    raw["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")
    raw["altitude_ft"] = pd.to_numeric(raw["altitude_ft"], errors="coerce")

    raw = raw.dropna(subset=["latitude", "longitude", "timestamp"])
    raw = raw[raw["latitude"].between(-90, 90) & raw["longitude"].between(-180, 180)]
    raw = raw.sort_values("timestamp").drop_duplicates(subset=["timestamp", "latitude", "longitude"])
    raw = raw.reset_index(drop=True)

    # Preserve the altitude column while converting feet to metres.
    raw["altitude_m"] = raw["altitude_ft"] * 0.3048
    return raw[["latitude", "longitude", "altitude_m", "timestamp"]]


def main() -> int:
    if not RAW_ROOT.is_dir():
        print(f"ERROR: GeoLife data folder not found:\n  {RAW_ROOT}")
        print("Check that the dataset is extracted under data/raw/Geolife Trajectories 1.3/Data/")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Remove old generated trip CSVs so stale results cannot be mistaken for this run.
    for old_file in OUTPUT_DIR.glob("trip_[0-9][0-9][0-9].csv"):
        old_file.unlink()
    if SUMMARY_PATH.exists():
        SUMMARY_PATH.unlink()

    files = sorted(RAW_ROOT.glob("*/Trajectory/*.plt"))
    if not files:
        print(f"ERROR: No .plt files found under:\n  {RAW_ROOT}")
        return 1

    print(f"Found {len(files):,} GeoLife trajectory files.")
    print(f"Extracting up to {MAX_TRAJECTORIES} valid trajectories (minimum {MIN_POINTS} points each)...")

    summary_rows = []
    skipped = 0
    saved = 0

    for index, path in enumerate(files, start=1):
        if saved >= MAX_TRAJECTORIES:
            break
        try:
            trip = read_geolife_plt(path)
            if len(trip) < MIN_POINTS:
                skipped += 1
                continue

            saved += 1
            out_path = OUTPUT_DIR / f"trip_{saved:03d}.csv"
            trip.to_csv(out_path, index=False)
            summary_rows.append({
                "trip_id": f"trip_{saved:03d}",
                "source_user_id": path.parent.parent.name,
                "source_file": path.name,
                "points": len(trip),
                "start_time": trip["timestamp"].iloc[0],
                "end_time": trip["timestamp"].iloc[-1],
                "csv_path": str(out_path.relative_to(PROJECT_ROOT)),
            })
            print(f"[{saved:02d}/{MAX_TRAJECTORIES}] Saved {out_path.name}: {len(trip):,} points")
        except (ValueError, TypeError, KeyError, pd.errors.ParserError, OSError) as exc:
            skipped += 1
            # Keep output concise; don't print hundreds of nearly identical failures.
            if skipped <= 3:
                print(f"Skipping unreadable file {path.name}: {exc}")
        except Exception as exc:
            skipped += 1
            if skipped <= 3:
                print(f"Skipping {path.name} ({type(exc).__name__}): {exc}")

        if index % 500 == 0:
            print(f"Scanned {index:,}/{len(files):,} files; saved {saved} valid trajectories...")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False)

    print("\nExtraction complete.")
    print(f"Valid trajectories saved: {saved}")
    print(f"Files skipped/too short: {skipped}")
    if summary_rows:
        print(f"Trip CSVs: {OUTPUT_DIR}")
        print(f"Summary:   {SUMMARY_PATH}")
    else:
        print("No valid trajectories were saved. Check the dataset path and .plt file format.")
        return 2

    print("\nNote: GeoLife contains mixed transport modes. These trajectories are not automatically confirmed as car/driving trips.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
