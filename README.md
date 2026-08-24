# AMQR

Code accompanying the manuscript on distance-driven, anchor-indexed quantile
regions for data supported on manifolds whose analytic geometry is unavailable.
The estimator uses only pairwise dissimilarities: it estimates intrinsic
geometry with a neighbourhood graph, constructs an intrinsic reference
measure, and aligns it to a Euclidean reference through anchored optimal
transport.  The resulting ranks define nested quantile regions around a
data-supported anchor.

The repository currently contains the reproducibility code for the manuscript
under review.  Raw datasets, generated results, and manuscript source files are
intentionally excluded.

## Repository layout

- `models/anchored_uniformization.py`: distance-only AMQR estimator and OOS
  rank extension.
- `models/hallin_liu.py`: analytic-geometry benchmark used on the sphere and
  flat torus.
- `experiments/`: canonical simulation, benchmark, real-data, and robustness
  entry points.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Run commands from the repository root.  For example, the main unconditional
and conditional simulations are reproduced with

```bash
python experiments/run_section4_2_manifold_regions.py
python experiments/run_section4_3_conditional_oos.py \
  --output-dir results/section4_3_conditional_oos_final
```

See [`experiments/README.md`](experiments/README.md) for all formal experiment
entry points.

## Data

Raw data are not distributed in this repository.  The NASA battery analysis
expects the downloaded dataset under
`data/raw/battery_alt_dataset/`, or a custom path supplied through the script's
command-line option.  Generated output is written below `results/`; both
directories are ignored by Git.

## Reproducibility status

Random seeds and output locations are exposed by the experiment scripts, and
formal runs write machine-readable manifests or summaries alongside figures.
The code package will remain private during anonymous review and can be
archived with a permanent release after acceptance.
