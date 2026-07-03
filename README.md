# Hybrid Photonic MI-BCI

Modular software baseline for a hybrid photonic EEG motor-imagery BCI
experiment on `BCICIV_1_asc`.

The central idea is not to treat the photonic chip as a one-off accelerator for
one fixed classifier. Instead, each EEG decision window scans a small bank of
candidate `2 x 8` calibration projections. The current implementation uses
NumPy matrix-vector multiplication, while the rest of the pipeline is already
structured around a backend interface that can later be replaced by photonic
hardware.

Current system outputs:

- `left`
- `right`
- `foot`
- `reject` / unrecognized

`left/right/foot` are learned MI classes. `reject` is a digital decision state,
not a training label.

## Repository Layout

```text
EMPT_MI_BCI/
  Dataset/
    BCICIV_1_asc/              Raw BCICIV_1_asc ASCII calibration files.
  hybrid_photonic_mi_bci/      Reusable Python package.
    datasets/                  BCICIV loader and feature extraction.
    backends.py                MVM backend interface and NumPy backend.
    experiment.py              Pipeline builders and replay helpers.
    pipeline.py                Online candidate-scan prediction flow.
    projection_library.py      Candidate projection library construction.
    decision.py                Prototype decision and reject logic.
    calibration.py             Online selector/fusion policies.
    features.py                Feature standardization utilities.
  examples/
    run_bciciv_replay.py       Main replay entry point.
  visualization/
    generate_bciciv_figures.py Figure generation entry point.
    README.md                  Figure script notes.
  artifacts/
    figures/bciciv_1_asc/      Generated PNG/PDF figures.
    README.md                  Artifact layout notes.
  tests/
    test_pipeline.py           Unit tests for the core pipeline.
```

Generated files are kept under `artifacts/`. The project root is intentionally
kept small and only contains source, tests, docs, and the active dataset folder.

## Environment

Python `>=3.10` is required.

Install the package with development and visualization dependencies:

```bash
python -m pip install -e ".[dev,viz]"
```

Minimal runtime dependencies are listed in `pyproject.toml`:

- `numpy`
- `scipy`
- optional: `matplotlib` for figures, `pytest` for development

## Dataset

The active dataset is:

```text
Dataset/BCICIV_1_asc/
```

The loader expects the original ASCII calibration files:

```text
BCICIV_calib_ds1a_cnt.txt
BCICIV_calib_ds1a_mrk.txt
BCICIV_calib_ds1a_nfo.txt
...
BCICIV_calib_ds1g_cnt.txt
BCICIV_calib_ds1g_mrk.txt
BCICIV_calib_ds1g_nfo.txt
```

BCICIV dataset 1 stores seven acquisition files, `a-g`. Each file is locally a
two-class motor-imagery recording, but the class pair differs by file. The
pooled loader reads each file's `nfo` metadata and maps local labels into the
global three-class set:

```text
left / right / foot
```

The default experiment pools all files into one replay problem. This matches the
current target system: three MI commands plus digital reject.

## Processing Pipeline

```text
BCICIV EEG trial
  -> common-average reference
  -> 8-30 Hz band-pass filtering
  -> 1.0-4.0 s decision window
  -> 8-D log-bandpower feature vector
  -> standardization from calibration trials
  -> candidate bank of N projection matrices, each 2 x 8
  -> MVMBackend.scan(weights, feature)
  -> prototype decision in 2-D candidate spaces
  -> probability fusion / online selector
  -> left, right, foot, or reject
```

Default feature channels:

```text
C3, C4, Cz, FC3, FC4, CP3, CP4, CPz
```

Default feature vector:

```text
8 log-variance features, one per selected channel
```

## Run Replay

Run the default pooled BCICIV replay:

```bash
python examples/run_bciciv_replay.py
```

Equivalent explicit command:

```bash
python examples/run_bciciv_replay.py --subject pooled
```

Run each acquisition file separately:

```bash
python examples/run_bciciv_replay.py --subject all
```

Run one file:

```bash
python examples/run_bciciv_replay.py --subject a
```

Useful options:

```bash
python examples/run_bciciv_replay.py --help
```

Common adjustable parameters:

- `--n-train`: calibration trials per file before replay.
- `--warmup-trials`: labeled windows used to warm up the selector.
- `--n-candidates`: number of candidate `2 x 8` projections scanned per trial.
- `--library-kind`: `perturb`, `bootstrap`, or `mixed`.
- `--selector`: `bandit`, `confidence`, or `fusion`.
- `--reject-threshold`: confidence threshold for reject.
- `--margin-threshold`: class-margin threshold for reject.
- `--channels`: comma-separated list of exactly eight EEG channels.
- `--band-low`, `--band-high`: band-pass range.
- `--window-start`, `--window-end`: trial window in seconds after marker.

## Default Experiment

Default pooled settings:

- dataset: `Dataset/BCICIV_1_asc`
- subjects/files: `a-g`, pooled
- classes: `left/right/foot`
- system outputs: `left/right/foot/reject`
- selected channels: `C3,C4,Cz,FC3,FC4,CP3,CP4,CPz`
- band: `8-30 Hz`
- window: `1.0-4.0 s`
- feature dimension: `8`
- training split: first `120` trials per file
- replay split: remaining `80` trials per file
- pooled warmup: `840` trials
- pooled replay: `560` trials
- candidate library: `32` candidate `2 x 8` projections
- selector: probability fusion

Recent default replay result:

```text
features: 1400 trials x 8 dims
warmup trials: 840
replay trials: 560
accepted accuracy: 0.665
command accuracy with rejects counted as misses: 0.664
balanced command accuracy: 0.650
reject rate: 0.002
confusion rows=true, cols=left/right/foot/reject:
[[190  46  44   0]
 [ 44 132  20   1]
 [ 21  12  50   0]]
```

These numbers are a reproducible baseline, not a final optimized MI decoder.
The feature extractor is intentionally simple so the photonic candidate-scan
boundary remains easy to inspect.

## Generate Figures

Generate all BCICIV figures:

```bash
python visualization/generate_bciciv_figures.py
```

Output directory:

```text
artifacts/figures/bciciv_1_asc/
```

Generated PNG/PDF figure pairs:

- `bciciv_system_block_diagram`: full system block diagram.
- `bciciv_feature_distributions`: feature distributions and class separability.
- `bciciv_projection_fit`: 2-D projection fit for the pooled classes.
- `bciciv_online_replay`: online accuracy, confidence, reject rate, and selector behavior.
- `bciciv_subject_summary`: per-file diagnostic summary.
- `bciciv_confusion_matrix`: pooled `left/right/foot/reject` confusion matrix.

## Run Tests

```bash
python -m unittest discover -s tests
```

If `pytest` is installed, this also works:

```bash
python -m pytest
```

## Module Guide

Important entry points:

- `hybrid_photonic_mi_bci.datasets.bciciv_1_asc.load_pooled_subject_features`
  loads `a-g`, maps local two-class labels into global `left/right/foot`, and
  returns fixed 8-D features.
- `hybrid_photonic_mi_bci.experiment.build_pipeline_from_features` builds the
  standardizer, candidate projection bank, decision head, selector, and backend.
- `hybrid_photonic_mi_bci.experiment.run_replay` runs labeled replay and returns
  metrics, predictions, confusion matrix, and per-window outputs.
- `examples.run_bciciv_replay.run_pooled` is the shortest programmatic path for
  reproducing the default experiment.

## Photonic Integration Point

All candidate projection scans pass through:

```python
MVMBackend.scan(weights, features) -> projections
```

Current software backend:

```python
NumpyMVMBackend
```

Future hardware backend placeholder:

```python
PhotonicMVMBackendStub
```

Contract:

```text
weights:  (n_candidates, 2, 8)
features: (8,)
return:   (n_candidates, 2)
```

That means a photonic implementation can be added behind the backend boundary
without changing feature extraction, candidate selection, reject logic, metrics,
or plotting code.

In practice, the hardware backend would hide details such as:

- weight programming / calibration
- feature quantization or scaling
- optical/electrical transport
- detector readout
- nonideality compensation
- batching or scan scheduling

As long as it returns the same `(n_candidates, 2)` projection array, the rest of
the software pipeline can remain unchanged.

## Current Scope

This repository currently contains a clean, real-data baseline:

- BCICIV_1_asc only
- pooled `a-g` loading by default
- three learned MI classes plus digital reject
- NumPy MVM backend with a photonic backend interface
- replay metrics and reproducible figures

Possible next technical extensions:

- stronger EEG features such as CSP or FBCSP
- reject calibration with a target reject/accuracy operating point
- cross-file validation summaries
- photonic nonideality simulation backend
- real photonic backend implementation
