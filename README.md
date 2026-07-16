# Hybrid Photonic MI-BCI

Modular implementation of a hybrid photonic motor-imagery BCI prototype on
`BCICIV_1_asc`.

The current design is not a log-bandpower shortcut. The implemented mainline is:

```text
FBCSP
  -> compact neural embedding
  -> experience-library retrieval
  -> multi-candidate linear-head scan through the MatrixOps/MVM backend
  -> backend probability fusion, digital softmax/reject
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
    BNCI2014_004/                 Local BCI IV 2b GDF dataset, ignored by git.
  hybrid_photonic_mi_bci/
    datasets/                     BCICIV loaders and pooled trial extraction.
    compute_accounting.py         Linear MAC accounting for photonic/digital split.
    fbcsp.py                      One-vs-rest filter-bank CSP.
    linear_models.py              Shrinkage LDA and linear-head helpers.
    small_networks.py             Compact FBCSP MLP embedding model.
    experience.py                 Experience-library retrieval and scan.
    backends.py                   Unified MatrixOps, SignalOps, and tiled MVM interfaces.
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
  -> 3rd-order filter bank: 8-12, 12-16, 16-20, 20-24, 24-28, 28-32 Hz
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
  -> candidate linear heads score_i = A_i [h, 1]
  -> TiledMVMBackend for augmented candidate heads
  -> MatrixOps probability fusion
  -> digital softmax and reject
```

The photonic primitive is modeled as a `2 x 8` tile, not an algorithmic size
limit. Larger matrices are scanned over row and column blocks:

```text
tile_count = K * ceil(M / tile_rows) * ceil(D_augmented / tile_cols)
```

For the default mainline:

```text
K = 8, M = 3, D_augmented = 32 + 1
tile_count = 8 * ceil(3/2) * ceil(33/8) = 80 tile evaluations/window
```

The extra dimension is the constant-one input used to scan linear-head bias on
the same photonic MVM path as the weights.

## Photonic Handoff Interface

Algorithm code should route matrix products and forward signal-processing
linear operations through `hybrid_photonic_mi_bci.backends` instead of calling
`np @`, `np.einsum`, or direct filtering kernels from workflow code. The default
matrix path is `AdaptivePrecisionPhotonicMatrixOpsBackend`. CAR starts at 4-bit
logical precision; SOS state transitions, FBCSP spatial projections, and feature
standardization start at 6-bit; MLP layers and other sensitive paths stay at
8-bit. The first call and periodic samples are checked against a digital 8-bit
bit-sliced shadow. An operation that exceeds its incremental-error limit is
immediately recomputed at the next precision and remains promoted. Every logical
precision is decomposed into radix-16 physical uint4/int4 slice pairs, while
system-level matrices are independently split into `2 x 8` spatial tiles.

The tiled candidate scan uses `QuantizedPhotonicMatrixOpsBackend` internally by
default and intentionally remains a single-pass 4-bit path. Both the candidate
scan and bit-sliced path use the same physical sweet-spot contract:

```text
qinmin = 0,  qinmax = 15
qwtmin = -8, qwtmax = 7
```

The physical tile executor follows the LT-Simulator `custom_matmul.py`
integration and formally runs through `osimulator.api.load_gazelle_model()`.
The integer NumPy path is retained only as a development fallback for interface
checks when the simulator is unavailable; its output is not a formal experiment
result. Simulator outputs produced by the experiment operator are synchronized
back into `artifacts/` for reporting, followed by zero-point correction and
dequantization or bit-slice reconstruction.

For example, an unsigned logical integer is expanded as
`q = q0 + 16*q1 + 16^2*q2 + ...`, where every `qi` is in `[0, 15]`. Signed
weights use balanced radix-16 digits in `[-8, 7]`. This preserves a higher
logical precision without requiring one physical call to exceed 4 bits. It is
runtime fixed-point decomposition, not QAT.

A real matrix driver can replace the matrix path with
`set_matrix_ops_backend(...)` or be scoped with `use_matrix_ops_backend(...)`.
A real signal-processing driver can replace CAR/SOS filtering with
`set_signal_ops_backend(...)` or `use_signal_ops_backend(...)`.

Main backend helpers:

```text
matrix_multiply / matrix_einsum
linear_scores
covariance_gram
csp_spatial_project
prototype_distances
candidate_probability_fusion
batched_matrix_vector
common_average_reference
signal_sosfiltfilt
```

The tiled candidate scan uses the same handoff internally, so the `2 x 8`
photonic primitive remains a hardware tile, not an algorithmic matrix-size
limit.

## Linear Compute Accounting

The project now reports the main photonic share on a forward-only basis. Model
fitting and training/backprop are excluded from this denominator; preprocessing,
calibration forward passes, and online inference remain included. Algorithm-path
matrix products routed through `MatrixOpsBackend` and forward CAR/SOS signal
operations routed through `SignalOpsBackend` are treated as photonic work under
the current software-substitution assumption:

```text
photonic_linear_share = forward_photonic_linear_MACs / forward_total_linear_MACs
```

Scope:

- counted as photonic: CAR expanded as a channel-mixing matrix, Butterworth SOS
  filtering expanded into per-section `3 x 3` state transitions for both filter
  directions, FBCSP covariance/projection matrix products,
  selected-feature standardization affine maps, LDA/linear-head scoring,
  exported MLP `Linear` layers, MLP LayerNorm affine parameters, linear-head
  and MLP bias via augmented constant-one inputs, experience-library retrieval
  distance cross terms, tiled candidate scan, retrieval-weight fusion, and
  prototype-distance cross terms in the legacy prototype path;
- counted as digital in the forward path: scalar control, thresholding, and
  other non-matrix/non-filter bookkeeping;
- excluded from this ratio: FBCSP/CSP fitting, LDA fitting, PyTorch compact-MLP
  training/backprop, LayerNorm mean/variance normalization, activations,
  softmax, remaining non-dot-product distance arithmetic, variance/log
  nonlinear feature operations, `eigh`/`pinv` decompositions, and
  visualization-only PCA/projection products.

CAR is executed as a fixed channel projection. Each SOS section is executed as
a `3 x 3` state transition whose coefficient MACs pass through the same
bit-sliced MatrixOps backend; forward and reverse passes retain zero-phase
filtering. Formal forward results use the Gazelle photonic simulator. These are
simulator results, not real-chip latency or power measurements.

The SOS filter estimate uses:

```text
MACs = trials * bands * channels * samples * sos_sections * 5 * 2
```

The final `* 2` is the forward/backward pass in `sosfiltfilt`. For the current
3rd-order Butterworth band-pass, `sos_sections = 3`. Small edge-padding
overheads are ignored. Earlier 4th-order runs increased this filtering MAC term
by about one third without improving the current validation metrics enough to
justify the cost. A 2nd-order run was also checked, but it degraded the BCICIV
mainline too much.

Saved accounting files:

```text
artifacts/metrics/fbcsp_design/compute_accounting.json
artifacts/metrics/bnci2014_004_personalization/compute_accounting.json
```

Run progress and timing files:

```text
artifacts/metrics/fbcsp_design/run_progress.json
artifacts/metrics/bnci2014_004_personalization/run_progress.json
```

Generated accounting figures:

```text
artifacts/figures/fbcsp_design/compute_accounting/compute_accounting_summary.png
artifacts/figures/bnci2014_004_personalization/bnci_compute_accounting_summary.png
```

## Run Experiments

Run the full comparison:

```bash
python examples/run_fbcsp_design_comparison.py
```

The command prints per-stage progress and elapsed time to the terminal and
saves the same records to `artifacts/metrics/fbcsp_design/run_progress.json`.
During the photonic-scan line, it also prints an online evaluation progress bar
with cumulative command accuracy, accepted accuracy, and reject rate.

Run each line separately:

```bash
python examples/run_fbcsp_reference.py
python examples/run_small_network_line.py
python examples/run_experience_photonic_line.py
```

Add `--no-progress` to the full comparison or photonic line script to suppress
the live terminal progress bar.

Run one full online inference pass from a single held-out EEG window:

```bash
python examples/run_single_window_inference.py --evaluation-index 0
```

This script explicitly separates offline setup from online execution. Dataset
loading, model fitting, bulk replay feature caching, and threshold calibration
use the NumPy/SciPy reference backend. After setup, the selected raw EEG window
runs FBCSP filtering/projection, standardization, compact-MLP inference, and the
candidate scan through the photonic backends. Terminal output reports elapsed
time, physical bit-sliced tile evaluations, selected precision, 8-bit-shadow
error, and escalation counts. Detailed per-operation telemetry is saved under
`artifacts/metrics/fbcsp_design/adaptive_precision_eval_XXXX.json`.

Useful parameters:

```bash
python examples/run_fbcsp_design_comparison.py --help
python examples/run_single_window_inference.py --help
```

Common knobs:

- `--selected-features`: selected FBCSP dimension, default `32`.
- `--filter-order`: Butterworth SOS band-pass order, default `3`.
- `--mlp-epochs`: compact MLP epochs, default `220`.
- `--experience-entries`: bootstrap experience entries, default `64`.
- `--experience-top-k`: scanned candidates, default `8`.
- `--calibration-trials-per-subject`: replay windows used for library query, default `6`.
- `--tile-rows`, `--tile-cols`: photonic tile shape, default `2 x 8`.
- `--online-repeats`: repeat one held-out window after a single setup and report median/P90 timing.
- `--precision-validation-windows`: compare adaptive and fixed-8-bit raw-window forward paths on N held-out windows.

## Current Results

Default results saved in `artifacts/metrics/fbcsp_design/summary.json`:

| Line | Eval windows | Command acc. | Balanced acc. | Reject rate | Forward MACs | Forward share | Inference share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FBCSP + shrinkage LDA | 560 | 0.693 | 0.713 | 0.016 | 0.576 GMAC | 1.000 | 1.000 |
| FBCSP + small MLP embedding | 560 | 0.723 | 0.709 | 0.075 | 0.584 GMAC | 1.000 | 1.000 |
| FBCSP + MLP embedding + library + photonic scan | 518 | 0.705 | 0.732 | 0.027 | 0.585 GMAC | 1.000 | 1.000 |

The mainline excludes the 42 calibration-query windows from its online
evaluation. The default 3rd-order filter is a compute/accuracy compromise:
`--filter-order 4` remains supported when accuracy is prioritized, while a
2nd-order run was too aggressive for the BCICIV mainline. Under the current
forward-only accounting, all measured forward linear MAC-equivalent work is
routed through MatrixOps or SignalOps handoff interfaces.

## BNCI2014_004 Three-Line Evaluation

The three implemented lines are also evaluated on `BNCI2014_004 / BCI
Competition IV 2b`. This dataset has 9 subjects, 5 sessions per subject, and 3
motor-area EEG channels (`C3`, `Cz`, `C4`) for left/right hand MI.

The implemented protocol uses only labeled `T` sessions:

```text
for each target subject:
  sessions 1-2 -> train/history set
  session 3    -> target new session
    first 8 trials/class -> mainline calibration query only
    remaining trials     -> shared held-out evaluation for all three lines
```

Run:

```bash
python examples/run_bnci2014_004_personalization.py
python visualization/plot_bnci2014_004_personalization.py
```

The BNCI run records one timed step per subject plus aggregation/saving in
`artifacts/metrics/bnci2014_004_personalization/run_progress.json`.

Default mean results across 9 subjects:

| Line | Eval windows | Command acc. | Balanced acc. | Reject rate | Forward MACs | Forward share | Inference share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FBCSP + shrinkage LDA | 1296 | 0.755 | 0.755 | 0.012 | 0.596 GMAC | 1.000 | 1.000 |
| FBCSP + small MLP embedding | 1296 | 0.744 | 0.744 | 0.014 | 0.599 GMAC | 1.000 | 1.000 |
| FBCSP + MLP + library + photonic scan | 1296 | 0.749 | 0.749 | 0.016 | 0.667 GMAC | 1.000 | 1.000 |

The BNCI workflow is binary MI, so it validates the three-line comparison and
experience-library retrieval behavior before returning to the four-output
`left/right/foot/reject` BCICIV setting. Its forward linear compute follows the
same MatrixOps + SignalOps handoff accounting.

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
artifacts/figures/fbcsp_design/compute_accounting/
artifacts/figures/fbcsp_design/adaptive_precision/
artifacts/figures/fbcsp_design/summary/
```

Key figures:

- `system_block_diagram_detailed`: concrete implementation block diagram.
- `reference_fbcsp_lda_diagnostics`: rolling accuracy/reject, confidence, confusion, feature selection.
- `small_network_training_embedding`: training curves, embedding PCA, replay trace, confusion.
- `experience_photonic_scan_diagnostics`: library weights, head quality, replay trace, confusion.
- `photonic_tile_schedule`: `2 x 8` tile work per decision window.
- `compute_accounting_summary`: forward MatrixOps/SignalOps-vs-digital linear MAC split and online share.
- `adaptive_precision_diagnostics`: selected bit widths, 8-bit-shadow errors, promotions, and tile use.
- `adaptive_vs_fixed8_validation`: preliminary task-metric, runtime, tile, and probability A/B comparison.
- `design_line_summary`: final line comparison.

## Precision and Hardware Roadmap

The current monitored policy is deliberately mixed:

- candidate-head scan: one physical uint4/int4 pass per spatial tile;
- CAR: 4-bit logical precision with 8-bit-shadow monitoring;
- SOS/FBCSP/feature standardization: 6-bit starting precision, promoted to
  8-bit per operation when sampled shadow error exceeds its limit;
- compact-MLP and other sensitive linear operators: 8-bit logical precision;
- matrix shape: independently tiled into `2 x 8` blocks;
- nonlinear activations, variance/log operations, sorting, rejection, and
  control flow: digital execution outside the linear-MAC denominator.

The next precision study must determine what each algorithm stage actually
needs instead of assigning one conservative precision everywhere. CAR, SOS
state transitions, FBCSP projection, standardization, individual MLP layers,
distance cross terms, candidate heads, and probability fusion should each be
swept across logical precision. The deployment policy should select the lowest
precision that still satisfies command accuracy, accepted accuracy, reject
rate, numerical stability, and noise-robustness targets. This per-operator mixed
precision policy can remove unnecessary slice pairs, tile evaluations, latency,
and energy.

A second study should widen the physical photonic unit from 4 bits to candidate
5/6/8-bit designs. Wider cells are expected to lose noise tolerance, but they
also reduce positional slices and partial-sum accumulation. The useful question
is therefore not whether wider precision is noisier in isolation, but whether
its measured accuracy and reject behavior remain inside engineering limits while
effective throughput, latency, and energy improve. Results should be reported as
a Pareto comparison over task metrics, injected hardware noise, effective bits,
physical tile/slice calls, latency, and energy.

Adaptive-vs-fixed-8-bit validation is executed on the Gazelle photonic
simulator. The simulator cannot be invoked from every development environment,
so the experiment operator runs it and synchronizes the resulting backend
identity, predictions, errors, tile/slice counts, and timings into `artifacts/`.
Current work-in-progress values from the three-window validation are:
`1.000` prediction agreement, equal `1.000` command accuracy, zero rejects,
`0.00775` mean probability L2 difference, `15.7%` fewer tile evaluations, and
median timings of `1.540 s` versus `1.634 s`. The repeated-window result reports
a `15.1%` steady-state tile reduction with `1.545 s` median and `1.600 s` P90
timing. These data and the precision policy are still being refined and are not
final conclusions. Subject-level conclusions require the complete replay.

## Tests

Run all tests:

```bash
python -m unittest discover -s tests
```

The tests cover spatial tile reconstruction, 8-bit logical values reconstructed
from physical uint4/int4 slices, signed candidate-head weights, photonic SOS
state-space equivalence against SciPy, generic MVM behavior, FBCSP output shapes,
shrinkage LDA, experience-library scan shape, replay split protocol, the Cyton
host simulator, Cyton command builder, and SQLite experience group store.
