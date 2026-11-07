"""Execute the preserved baseline without touching historical results."""
import argparse
from pathlib import Path
import run_experiment_multitrip as baseline
p=argparse.ArgumentParser(); p.add_argument('--output',default='results/baseline_new'); a=p.parse_args()
out=Path(a.output)
if out.exists(): raise FileExistsError('Choose a new baseline output directory')
baseline.RESULTS=out; baseline.TABLES=out/'tables'; baseline.FIGURES=out/'figures'
baseline.main()
