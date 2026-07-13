# Visualization Scripts

The current design figures are generated from saved FBCSP design metrics.
Plotting scripts do not retrain models.

Generate all current figures from the project root:

```bash
python visualization/generate_fbcsp_design_figures.py
```

Default output:

```text
artifacts/figures/fbcsp_design/
```

Figure modules:

```text
visualization/fbcsp_design/
  plot_system_diagram.py
  plot_reference.py
  plot_small_network.py
  plot_experience_photonic.py
  plot_compute_accounting.py
  plot_summary.py
```

Generated figure groups:

- `system`: detailed implementation block diagram.
- `reference`: FBCSP + shrinkage LDA diagnostics.
- `small_network`: compact MLP training and embedding diagnostics.
- `experience_photonic`: experience retrieval and tiled candidate-scan diagnostics.
- `compute_accounting`: forward MatrixOps/SignalOps-vs-digital linear MAC split.
- `summary`: final line comparison.

The older `generate_bciciv_figures.py` script is retained for the legacy
log-bandpower ablation path only.

For the BNCI2014_004 three-line comparison:

```bash
python visualization/plot_bnci2014_004_personalization.py
```

Default output:

```text
artifacts/figures/bnci2014_004_personalization/
```
