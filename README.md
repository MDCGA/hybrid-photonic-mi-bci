# Hybrid Photonic MI-BCI

Modular implementation of a hybrid photonic motor-imagery BCI prototype on
`BCICIV_1_asc`.

The current design is not a log-bandpower shortcut. The implemented mainline is:

```text
FBCSP
  -> compact neural embedding
  -> experience-library retrieval
  -> multi-candidate linear-head scan through an MVM backend
  -> digital fusion, softmax, and reject
```

The traditional reference baseline is:

```text
FBCSP + shrinkage LDA
```

The fourth output, `reject`, is a digital decision state. The learned MI classes
are `left`, `right`, and `foot`.

## Repository Layout

```text
EMPT_MI_BCI/
  Dataset/
    BCICIV_1_asc/                 Local raw ASCII dataset, ignored by git.
  hybrid_photonic_mi_bci/
    datasets/                     BCICIV loaders and pooled trial extraction.
    fbcsp.py                      One-vs-rest filter-bank CSP.
    linear_models.py              Shrinkage LDA and linear-head helpers.
    small_networks.py             Compact FBCSP MLP embedding model.
    experience.py                 Experience-library retrieval and scan.
    backends.py                   NumPy and tiled MVM backend interfaces.
    host/                         Cyton host app, acquisition, and library store.
    workflows/                    Clean experiment lines and shared protocol.
  examples/
    run_cyton_host.py             Cyton host CLI/Tkinter entry point.
    run_fbcsp_reference.py
    run_small_network_line.py
    run_experience_photonic_line.py
    run_fbcsp_design_comparison.py
  visualization/
    fbcsp_design/                 Split plotting modules.
    generate_fbcsp_design_figures.py
  artifacts/
    metrics/fbcsp_design/         Saved JSON/NPZ results.
    figures/fbcsp_design/         Generated PNG/PDF figures.
  tests/
```

Legacy log-bandpower replay code is kept only as a low-complexity ablation/debug
path. It is not the main baseline for the current design.

## Environment

Python `>=3.10` is required.

For the full FBCSP design, install development, plotting, and torch extras:

```bash
python -m pip install -e ".[dev,viz,torch]"
```

Core dependencies are declared in `pyproject.toml`. The compact embedding line
uses PyTorch. The reference FBCSP-LDA line can be imported without eagerly
loading torch.

For OpenBCI Cyton hardware control through BrainFlow:

```bash
python -m pip install -e ".[cyton]"
```

## Dataset

Expected local dataset path:

```text
Dataset/BCICIV_1_asc/
```

Required files:

```text
BCICIV_calib_ds1a_cnt.txt
BCICIV_calib_ds1a_mrk.txt
BCICIV_calib_ds1a_nfo.txt
...
BCICIV_calib_ds1g_cnt.txt
BCICIV_calib_ds1g_mrk.txt
BCICIV_calib_ds1g_nfo.txt
```

BCICIV dataset 1 contains seven acquisition files, `a-g`. Each file is a local
two-class recording, but the class pair differs by file. The pooled loader reads
each file's `nfo` metadata and maps labels into:

```text
left / right / foot
```

The ASCII marker convention used here is:

```text
mrk.y = -1 -> first class in nfo.classes
mrk.y = +1 -> second class in nfo.classes
```

Default split:

```text
train: 120 trials/file = 840 trials
replay: 80 trials/file = 560 trials
mainline calibration query: first 6 replay trials/file = 42 trials
mainline online evaluation: remaining 518 replay trials
```

## Implemented Pipeline

Shared FBCSP feature layer:

```text
EEG trial
  -> common-average reference
  -> selected motor channels
  -> marker + 1.0-4.0 s window
  -> filter bank: 8-12, 12-16, 16-20, 20-24, 24-28, 28-32 Hz
  -> one-vs-rest CSP for left/right/foot
  -> log-variance features
  -> 72D raw FBCSP vector
  -> Fisher feature selection, default 32D
  -> train-fitted standardization
```

Reference line:

```text
FBCSP 32D -> shrinkage LDA -> softmax/reject
```

Embedding line:

```text
FBCSP 32D
  -> LayerNorm
  -> Linear(32, 64) + GELU + Dropout
  -> Linear(64, 32) + GELU
  -> embedding h
  -> classifier logits
```

Mainline:

```text
embedding h
  -> experience library with bootstrap linear heads
  -> global anchor heads: MLP classifier and embedding-LDA
  -> calibration-aware top-K retrieval, default K=8
  -> candidate linear heads score_i = A_i h + b_i
  -> TiledMVMBackend for A_i h
  -> digital bias, softmax, fusion, and reject
```

The photonic primitive is modeled as a `2 x 8` tile, not an algorithmic size
limit. Larger matrices are scanned over row and column blocks:

```text
tile_count = K * ceil(M / 2) * ceil(D / 8)
```

For the default mainline:

```text
K = 8, M = 3, D = 32
tile_count = 8 * ceil(3/2) * ceil(32/8) = 64 tile evaluations/window
```

## Run Experiments

Run the full comparison:

```bash
python examples/run_fbcsp_design_comparison.py
```

Run each line separately:

```bash
python examples/run_fbcsp_reference.py
python examples/run_small_network_line.py
python examples/run_experience_photonic_line.py
```

Useful parameters:

```bash
python examples/run_fbcsp_design_comparison.py --help
```

Common knobs:

- `--selected-features`: selected FBCSP dimension, default `32`.
- `--mlp-epochs`: compact MLP epochs, default `220`.
- `--experience-entries`: bootstrap experience entries, default `64`.
- `--experience-top-k`: scanned candidates, default `8`.
- `--calibration-trials-per-subject`: replay windows used for library query, default `6`.
- `--tile-rows`, `--tile-cols`: photonic tile shape, default `2 x 8`.

## Current Results

Default results saved in `artifacts/metrics/fbcsp_design/summary.json`:

| Line | Eval windows | Command acc. | Balanced acc. | Accepted acc. | Reject rate | Tile evals/window |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FBCSP + shrinkage LDA | 560 | 0.713 | 0.732 | 0.723 | 0.014 | 0 |
| FBCSP + small MLP embedding | 560 | 0.743 | 0.718 | 0.797 | 0.068 | 0 |
| FBCSP + MLP embedding + library + photonic scan | 518 | 0.751 | 0.729 | 0.783 | 0.041 | 64 |

The mainline excludes the 42 calibration-query windows from its online
evaluation.

## Experience-Library Personalization Test

To directly test whether the experience library helps a single target subject
specialize, use `BNCI2014_004 / BCI Competition IV 2b`. This dataset has 9
subjects, 5 sessions per subject, and 3 motor-area EEG channels (`C3`, `Cz`,
`C4`) for left/right hand MI.

The implemented protocol uses only labeled `T` sessions:

```text
for each target subject:
  sessions 1-2 -> subject history / experience library
  session 3    -> target new session
    first k trials/class -> target calibration
    remaining trials     -> held-out evaluation
```

Run:

```bash
python examples/run_bnci2014_004_personalization.py
python visualization/plot_bnci2014_004_personalization.py
```

Default mean results across 9 subjects:

| k trials/class | Before personalization | Calibration only | Experience + calibration | Mean gain vs before | Improved subjects |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.756 | 0.614 | 0.766 | +0.011 | 6/9 |
| 4 | 0.756 | 0.703 | 0.764 | +0.008 | 6/9 |
| 8 | 0.755 | 0.738 | 0.773 | +0.018 | 9/9 |
| 12 | 0.763 | 0.750 | 0.779 | +0.016 | 7/9 |
| 16 | 0.761 | 0.760 | 0.775 | +0.014 | 6/9 |

This is the first direct evidence in the repository that the experience-library
mechanism can improve held-out performance after a small amount of target-session
calibration. The current test is binary MI, so it validates personalization
behavior before returning to the four-output `left/right/foot/reject` system.

## Cyton Host Application

The host-app layer is separate from the offline BCICIV workflows. It is the
starting point for the OpenBCI Cyton upper-computer software:

```text
Cyton / synthetic Cyton
  -> acquisition adapter
  -> host controller
  -> SQLite experience-library groups
  -> calibration/session artifacts
  -> future online FBCSP + candidate-scan runtime
```

Modules:

```text
hybrid_photonic_mi_bci/host/
  acquisition.py          BrainFlow Cyton adapter, synthetic source, Cyton commands.
  controller.py           Device/store controller used by CLI or GUI.
  experience_store.py     SQLite group and entry management.
  tk_app.py               Lightweight Tkinter operator UI.
```

Use the simulator first:

```bash
python examples/run_cyton_host.py --db artifacts/tmp/cyton_host_demo.sqlite init-db
python examples/run_cyton_host.py --db artifacts/tmp/cyton_host_demo.sqlite stream --synthetic --duration 2
python examples/run_cyton_host.py --db artifacts/tmp/cyton_host_demo.sqlite gui
```

For real Cyton hardware:

```bash
python examples/run_cyton_host.py --db artifacts/host/cyton_experience.sqlite stream --serial-port COM3 --duration 5
```

Replace `COM3` with the serial port of the OpenBCI dongle. Local host databases
and live acquisition exports are ignored under `artifacts/host/`.

The current store supports:

- experience-library group creation/listing/activation;
- per-group device, channel set, preprocessing, FBCSP, and encoder metadata;
- entry metadata for future saved CSP filters, selected features, embeddings,
  candidate heads, reject thresholds, and performance metrics.

## Figures

Regenerate all FBCSP design figures:

```bash
python visualization/generate_fbcsp_design_figures.py
```

Generated figure groups:

```text
artifacts/figures/fbcsp_design/system/
artifacts/figures/fbcsp_design/reference/
artifacts/figures/fbcsp_design/small_network/
artifacts/figures/fbcsp_design/experience_photonic/
artifacts/figures/fbcsp_design/summary/
```

Key figures:

- `system_block_diagram_detailed`: concrete implementation block diagram.
- `reference_fbcsp_lda_diagnostics`: rolling accuracy/reject, confidence, confusion, feature selection.
- `small_network_training_embedding`: training curves, embedding PCA, replay trace, confusion.
- `experience_photonic_scan_diagnostics`: library weights, head quality, replay trace, confusion.
- `photonic_tile_schedule`: `2 x 8` tile work per decision window.
- `design_line_summary`: final line comparison.

## Tests

Run all tests:

```bash
python -m unittest discover -s tests
```

The tests cover the generic MVM backend, tiled MVM behavior, FBCSP output shapes,
shrinkage LDA, experience-library scan shape, replay split protocol, the Cyton
host simulator, Cyton command builder, and SQLite experience group store.
