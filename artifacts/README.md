# Artifacts

Generated outputs are stored here to keep the project root clean.

Current layout:

```text
artifacts/
  metrics/
    fbcsp_design/
      summary.json
      reference/
      small_network/
      experience_photonic/
  figures/
    fbcsp_design/
      system/
      reference/
      small_network/
      experience_photonic/
      summary/
    bciciv_1_asc/        Legacy log-bandpower ablation figures, if generated.
```

Regenerate the current FBCSP design metrics:

```bash
python examples/run_fbcsp_design_comparison.py
```

Regenerate the current FBCSP design figures:

```bash
python visualization/generate_fbcsp_design_figures.py
```
