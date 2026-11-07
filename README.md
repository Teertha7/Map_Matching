# Map matching under GNSS degradation

Evaluate map matching on GeoLife GPS trajectories using an existing OpenStreetMap road network. Road labels are nearest-clean-GPS proxies, not verified ground truth.

## Project contents

- `src/`: experiment, data preparation, analysis, and validation code.
- `data/raw/`: original GeoLife dataset and its user guide.
- `data/processed/`: road network, selected trajectory, original trips, and vehicle cohort.
- `configs/`: extension experiment protocol.
- `tests/`: experiment unit tests.
- `requirements.txt`: pinned experiment dependencies.
- `.venv/`: existing local Python environment.

Run commands from this project directory. The prepared datasets are already available; no downloads or preparation steps are needed for normal experiments.

## Run the primary experiment

The extension uses 54 car/taxi journeys from 18 users with separate training and held-out users.

```sh
MPLCONFIGDIR=/tmp/mapmatching-mpl .venv/bin/python -m unittest discover -s tests -v
MPLCONFIGDIR=/tmp/mapmatching-mpl .venv/bin/python src/extended_experiment.py --smoke --output results/extension_smoke
MPLCONFIGDIR=/tmp/mapmatching-mpl .venv/bin/python src/extended_experiment.py --output results/extension
```

Use a new output directory for each run. Experiments recreate `results/` and save trained models, predictions, metrics, and metadata there. Previous outputs and document/report artifacts have been removed.

Optional analysis after the full extension run:

```sh
MPLCONFIGDIR=/tmp/mapmatching-mpl .venv/bin/python src/analyze_extension.py --input results/extension
.venv/bin/python src/extension_cv.py
.venv/bin/python src/verify_extension.py
```

The CV and verification scripts expect a full run at `results/extension`.

## Original experiment and baseline

```sh
MPLCONFIGDIR=/tmp/mapmatching-mpl .venv/bin/python src/research_experiment.py --smoke --output results/original_smoke
MPLCONFIGDIR=/tmp/mapmatching-mpl .venv/bin/python src/research_experiment.py --output results/final
MPLCONFIGDIR=/tmp/mapmatching-mpl .venv/bin/python src/reproduce_baseline.py --output results/baseline
```

`training_cross_validation.py` expects a full original run at `results/final`. The original cohort contains one user's trips and does not demonstrate generalization to unseen users.

## Environment

Use the existing `.venv`, or install `requirements.txt` in a compatible Python environment. The optional road-network download script additionally requires `osmnx`; it is unnecessary when using the included road network.
# Map_Matching
