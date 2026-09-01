# Paper experiment entry points

The scripts in this directory are the canonical reproducibility entry points
for the Adaptive Manifold Quantile Regions (AMQR) manuscript.  Uniformization
is fitted without an anchor, after which the fitted transport yields an
anchor-indexed family of regions.  Retired legacy GW-radialization, per-anchor
hard-transport, and exploratory data-analysis drivers have been removed.  Raw
data are never modified by these scripts; every entry point writes to a
dedicated directory under `results/`.

Run all commands from the repository root with a Python environment containing
the packages in `requirements.txt`.

## Main paper

| Manuscript output | Script | Default result directory |
|---|---|---|
| Section 4.2, cross-manifold recovery and component diagnostics | `run_section4_2_manifold_regions.py` | `results/section4_2_manifold_regions/` |
| Section 4.2, known-sphere benchmark | `run_hallin_liu_sphere_pilot.py` | `results/hallin_liu_sphere_pilot/` |
| Section 4.2, known-flat-torus benchmark | `run_hallin_liu_torus_pilot.py` | `results/hallin_liu_torus_pilot/` |
| Section 4.3, conditional and OOS recovery | `run_section4_3_conditional_oos.py` | supply `results/section4_3_conditional_oos_final/` |
| Section 4.3, conditional known-sphere benchmark | `run_hallin_liu_conditional_sphere.py` | `results/hallin_liu_conditional_sphere/` |
| Section 5, NASA battery validation | `run_real_nasa_battery_uniformization.py` | `results/real_nasa_battery_uniformization/` |

## Supplementary Appendix B

| Appendix output | Script | Default result directory |
|---|---|---|
| Entropic-regularization sensitivity | `run_anchor_indexed_entropy_sensitivity.py` | `results/anchor_indexed_entropy_sensitivity/` |
| Anchor-pool and fixed-support Bayesian-bootstrap stability | `run_anchor_pool_bootstrap_stability.py` | `results/anchor_pool_bootstrap_stability/` |

`_paper_simulation_utils.py` is a support module, not an experiment.  It holds
only analytic simulation geometry and the deterministic neighbourhood rules
reported in Section 4.1.  These oracle parameterizations are not passed to the
distance-only estimator.

## Reproduction commands

```bash
python -B experiments/run_section4_2_manifold_regions.py
python -B experiments/run_hallin_liu_sphere_pilot.py
python -B experiments/run_hallin_liu_torus_pilot.py
python -B experiments/run_section4_3_conditional_oos.py --output-dir results/section4_3_conditional_oos_final
python -B experiments/run_hallin_liu_conditional_sphere.py
python -B experiments/run_real_nasa_battery_uniformization.py
python -B experiments/run_anchor_indexed_entropy_sensitivity.py
python -B experiments/run_anchor_pool_bootstrap_stability.py
```

Each formal result directory contains machine-readable records and either a
`run_manifest.json`, `summary.json`, or experiment-specific JSON summary.  The
Appendix B numerical audit is also summarized in
`results/SUPPLEMENTARY_AUDITS_REDO_SUMMARY.md`.
