# Adaptive Manifold Quantile Regions (AMQR)

Code accompanying the manuscript on Adaptive Manifold Quantile Regions for
data supported on manifolds whose analytic geometry is unavailable.  AMQR
learns latent metric--measure geometry and corrects sampling density using only
pairwise dissimilarities.  It estimates intrinsic geometry with a neighbourhood
graph, constructs an intrinsic reference measure, and aligns it through optimal
transport.  The resulting ranks define an anchor-indexed family of nested
quantile regions around a data-supported reference point.

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
uses *An accelerated Life Testing Dataset for Lithium-Ion Batteries with
Constant and Variable Loading Conditions*, version 0.0.1 (Fricke, Nascimento,
and Viana, 2023).  Place its three folders, `regular_alt_batteries/`,
`recommissioned_batteries/`, and `second_life_batteries/`, under
`data/raw/battery_alt_dataset/`, or pass a different location with
`--dataset-dir`.  Generated output is written below `results/`; both the raw
data and generated results are ignored by Git.

## Reproducibility status

Random seeds and output locations are exposed by the experiment scripts, and
formal runs write machine-readable manifests or summaries alongside figures.
The review snapshot is intended for anonymous reproducibility and can be
archived as a versioned release after acceptance.
