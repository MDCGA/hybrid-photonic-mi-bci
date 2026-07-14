# Pure Runtime

This folder is the clean deployment-facing slice of the project. It deliberately
excludes dataset loaders, experiment comparison code, metrics, visualization,
tests, and saved artifacts.

It assumes the upstream signal path has already produced embeddings `h` and an
experience library of candidate linear heads. The runtime only does:

```text
calibration embeddings
  -> retrieve top-K experience entries
online embeddings
  -> 4-bit quantized tiled photonic candidate scan
  -> probability fusion
  -> command/reject decision
```

The photonic scan uses the same 4-bit profile as the current hardware sweet
spot:

```text
qinmin = 0,  qinmax = 15
qwtmin = -8, qwtmax = 7
```

Minimal use:

```python
from pure_runtime import PurePhotonicScanRuntime

runtime = PurePhotonicScanRuntime(
    entries=experience_entries,
    class_names=("left", "right", "foot"),
)
runtime.calibrate(calibration_embeddings)
outputs = runtime.predict(online_embeddings)
```

`experience_entries` are `hybrid_photonic_mi_bci.experience.ExperienceEntry`
objects. The folder is intentionally small; calibration/training and artifact
serialization should live outside this pure runtime layer.
