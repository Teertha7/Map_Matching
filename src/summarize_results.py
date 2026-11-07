
from pathlib import Path
import pandas as pd

PROJECT = Path.home() / "Documents" / "MapMatchingProject"
METRICS = PROJECT / "results" / "tables" / "experiment_metrics.csv"
OUTPUT = PROJECT / "results" / "tables" / "results_summary.txt"

def main():
    if not METRICS.exists():
        raise FileNotFoundError(f"Metrics file not found: {METRICS}")

    df = pd.read_csv(METRICS)

    print("\nEXPERIMENT METRICS")
    print(df.round(4).to_string(index=False))

    lines = [
        "ROBUST ML MAP-MATCHING: INITIAL EXPERIMENT",
        "=" * 48,
        "",
        "Evaluation target: nearest road to the original GPS coordinate.",
        "Interpretation: proxy-label agreement, not verified road truth.",
        "",
        "Results:"
    ]

    for _, row in df.iterrows():
        lines.append(
            f"Noise {row['noise_m']:.0f} m: "
            f"nearest accuracy={row['nearest_accuracy']:.4f}, "
            f"RF accuracy={row['rf_accuracy']:.4f}, "
            f"nearest macro-F1={row['nearest_macro_f1']:.4f}, "
            f"RF macro-F1={row['rf_macro_f1']:.4f}, "
            f"points={int(row['evaluated_points'])}"
        )

    lines += [
        "",
        "Limitations:",
        "- Targets are proxy labels derived from original GPS coordinates.",
        "- Road-level ground truth has not been independently verified.",
        "- This experiment does not establish highway/service-road accuracy.",
        "- Results use one selected trajectory and simulated coordinate noise."
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved summary to: {OUTPUT}")

if __name__ == "__main__":
    main()
