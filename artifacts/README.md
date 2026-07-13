# Artifacts

Generated outputs are stored here to keep the project root clean.

Current layout:

```text
artifacts/
  metrics/
    bnci2014_004_personalization/
      summary.json
      compute_accounting.json
      subject_rows.json
      arrays.npz
    fbcsp_design/
      summary.json
      compute_accounting.json
      reference/
      small_network/
      experience_photonic/
  figures/
    bnci2014_004_personalization/
    fbcsp_design/
      system/
      reference/
      small_network/
      experience_photonic/
      compute_accounting/
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

Regenerate the BNCI2014_004 three-line comparison:

```bash
python examples/run_bnci2014_004_personalization.py
python visualization/plot_bnci2014_004_personalization.py
```
