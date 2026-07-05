"""BCI Competition IV dataset 1 ASCII loader and 8-D feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]

DEFAULT_MOTOR_CHANNELS = ("C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4", "CPz")
DEFAULT_BAND = (8.0, 30.0)
DEFAULT_WINDOW = (1.0, 4.0)
DEFAULT_SUBJECTS = tuple("abcdefg")


@dataclass(frozen=True)
class BCICIVRecording:
    """One labeled calibration recording from BCICIV_1_asc."""

    subject: str
    fs: float
    class_names: tuple[str, ...]
    channel_names: tuple[str, ...]
    samples: FloatArray
    marker_positions: IntArray
    marker_codes: IntArray


@dataclass(frozen=True)
class BCICIVFeatures:
    """Trial-level features derived from one BCICIV calibration recording."""

    subject: str
    features: FloatArray
    labels: IntArray
    class_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    marker_positions: IntArray
    window: tuple[float, float]
    band: tuple[float, float]


@dataclass(frozen=True)
class BCICIVTrials:
    """Windowed trial epochs for feature extractors such as FBCSP."""

    subject: str
    trials: FloatArray
    labels: IntArray
    class_names: tuple[str, ...]
    channel_names: tuple[str, ...]
    marker_positions: IntArray
    fs: float
    window: tuple[float, float]


def load_calibration_recording(
    data_dir: str | Path,
    subject: str = "a",
) -> BCICIVRecording:
    """Load one BCICIV calibration recording.

    The ASCII calibration files contain continuous EEG samples (`cnt`), trial
    markers and labels (`mrk`), and metadata (`nfo`).
    """

    data_path = Path(data_dir)
    subject = _normalize_subject(subject)
    prefix = f"BCICIV_calib_ds1{subject}"
    nfo = _parse_nfo(data_path / f"{prefix}_nfo.txt")
    samples = np.loadtxt(data_path / f"{prefix}_cnt.txt", dtype=np.float64)
    markers = np.loadtxt(data_path / f"{prefix}_mrk.txt", dtype=np.float64)
    if markers.ndim != 2 or markers.shape[1] != 2:
        raise ValueError(f"marker file must have shape (N, 2), got {markers.shape}")

    return BCICIVRecording(
        subject=subject,
        fs=float(nfo["fs"]),
        class_names=tuple(nfo["classes"]),
        channel_names=tuple(nfo["clab"]),
        samples=samples,
        marker_positions=np.rint(markers[:, 0]).astype(int),
        marker_codes=np.rint(markers[:, 1]).astype(int),
    )


def load_subject_features(
    data_dir: str | Path,
    subject: str = "a",
    channels: Iterable[str] = DEFAULT_MOTOR_CHANNELS,
    band: tuple[float, float] = DEFAULT_BAND,
    window: tuple[float, float] = DEFAULT_WINDOW,
) -> BCICIVFeatures:
    """Load one subject and extract default 8-D log-bandpower features."""

    recording = load_calibration_recording(data_dir=data_dir, subject=subject)
    return extract_log_bandpower_features(
        recording=recording,
        channels=channels,
        band=band,
        window=window,
    )


def load_pooled_subject_features(
    data_dir: str | Path,
    subjects: Iterable[str] = DEFAULT_SUBJECTS,
    n_train_per_subject: int = 120,
    channels: Iterable[str] = DEFAULT_MOTOR_CHANNELS,
    band: tuple[float, float] = DEFAULT_BAND,
    window: tuple[float, float] = DEFAULT_WINDOW,
) -> BCICIVFeatures:
    """Load multiple BCICIV files as one pooled three-class dataset.

    Each acquisition file contains two labeled MI classes, but the class pair
    differs across files. The pooled dataset maps local labels to the global
    classes `left`, `right`, and `foot`. The fourth system output is the digital
    reject state, not a training label.

    The returned trial order is all per-subject training trials first, followed
    by all per-subject replay trials. This lets a single `n_train` boundary
    represent a fair pooled train/replay split.
    """

    subject_list = tuple(_normalize_subject(subject) for subject in subjects)
    if not subject_list:
        raise ValueError("at least one subject is required")

    global_class_names = ("left", "right", "foot")
    global_class_to_index = {name: index for index, name in enumerate(global_class_names)}
    train_features = []
    train_labels = []
    train_positions = []
    replay_features = []
    replay_labels = []
    replay_positions = []
    feature_names: tuple[str, ...] | None = None
    for subject in subject_list:
        subject_features = load_subject_features(
            data_dir=data_dir,
            subject=subject,
            channels=channels,
            band=band,
            window=window,
        )
        if not 0 < n_train_per_subject < len(subject_features.labels):
            raise ValueError(
                "n_train_per_subject must leave replay samples for every subject, "
                f"got {n_train_per_subject}"
            )
        if feature_names is None:
            feature_names = subject_features.feature_names
        elif feature_names != subject_features.feature_names:
            raise ValueError("all pooled subjects must use the same feature names")

        train_slice = slice(0, n_train_per_subject)
        replay_slice = slice(n_train_per_subject, None)
        mapped_labels = _map_local_to_global_labels(
            subject_features.labels,
            subject_features.class_names,
            global_class_to_index,
        )
        train_features.append(subject_features.features[train_slice])
        train_labels.append(mapped_labels[train_slice])
        train_positions.append(subject_features.marker_positions[train_slice])
        replay_features.append(subject_features.features[replay_slice])
        replay_labels.append(mapped_labels[replay_slice])
        replay_positions.append(subject_features.marker_positions[replay_slice])

    return BCICIVFeatures(
        subject=f"pooled_{''.join(subject_list)}",
        features=np.concatenate([*train_features, *replay_features], axis=0),
        labels=np.concatenate([*train_labels, *replay_labels], axis=0),
        class_names=global_class_names,
        feature_names=feature_names or tuple(),
        marker_positions=np.concatenate([*train_positions, *replay_positions], axis=0),
        window=window,
        band=band,
    )


def load_subject_trials(
    data_dir: str | Path,
    subject: str = "a",
    channels: Iterable[str] = DEFAULT_MOTOR_CHANNELS,
    window: tuple[float, float] = DEFAULT_WINDOW,
) -> BCICIVTrials:
    """Load one subject and return windowed EEG trial epochs.

    The returned trials have shape ``(n_trials, n_channels, n_samples)``.
    Common-average reference is computed on all available channels before
    selecting the requested channel subset.
    """

    recording = load_calibration_recording(data_dir=data_dir, subject=subject)
    return extract_trials(recording=recording, channels=channels, window=window)


def load_pooled_subject_trials(
    data_dir: str | Path,
    subjects: Iterable[str] = DEFAULT_SUBJECTS,
    n_train_per_subject: int = 120,
    channels: Iterable[str] = DEFAULT_MOTOR_CHANNELS,
    window: tuple[float, float] = DEFAULT_WINDOW,
) -> BCICIVTrials:
    """Load multiple BCICIV files as one pooled three-class trial dataset.

    The returned order matches :func:`load_pooled_subject_features`: all
    per-subject training trials first, then all per-subject replay trials.
    """

    subject_list = tuple(_normalize_subject(subject) for subject in subjects)
    if not subject_list:
        raise ValueError("at least one subject is required")

    global_class_names = ("left", "right", "foot")
    global_class_to_index = {name: index for index, name in enumerate(global_class_names)}
    train_trials = []
    train_labels = []
    train_positions = []
    replay_trials = []
    replay_labels = []
    replay_positions = []
    channel_names: tuple[str, ...] | None = None
    fs: float | None = None
    for subject in subject_list:
        subject_trials = load_subject_trials(
            data_dir=data_dir,
            subject=subject,
            channels=channels,
            window=window,
        )
        if not 0 < n_train_per_subject < len(subject_trials.labels):
            raise ValueError(
                "n_train_per_subject must leave replay samples for every subject, "
                f"got {n_train_per_subject}"
            )
        if channel_names is None:
            channel_names = subject_trials.channel_names
        elif channel_names != subject_trials.channel_names:
            raise ValueError("all pooled subjects must use the same channels")
        if fs is None:
            fs = subject_trials.fs
        elif fs != subject_trials.fs:
            raise ValueError("all pooled subjects must use the same sampling rate")

        mapped_labels = _map_local_to_global_labels(
            subject_trials.labels,
            subject_trials.class_names,
            global_class_to_index,
        )
        train_slice = slice(0, n_train_per_subject)
        replay_slice = slice(n_train_per_subject, None)
        train_trials.append(subject_trials.trials[train_slice])
        train_labels.append(mapped_labels[train_slice])
        train_positions.append(subject_trials.marker_positions[train_slice])
        replay_trials.append(subject_trials.trials[replay_slice])
        replay_labels.append(mapped_labels[replay_slice])
        replay_positions.append(subject_trials.marker_positions[replay_slice])

    return BCICIVTrials(
        subject=f"pooled_{''.join(subject_list)}",
        trials=np.concatenate([*train_trials, *replay_trials], axis=0),
        labels=np.concatenate([*train_labels, *replay_labels], axis=0),
        class_names=global_class_names,
        channel_names=channel_names or tuple(),
        marker_positions=np.concatenate([*train_positions, *replay_positions], axis=0),
        fs=float(fs or 0.0),
        window=window,
    )


def extract_trials(
    recording: BCICIVRecording,
    channels: Iterable[str] = DEFAULT_MOTOR_CHANNELS,
    window: tuple[float, float] = DEFAULT_WINDOW,
) -> BCICIVTrials:
    """Extract CAR-referenced trial epochs for the selected channels."""

    selected_channels = tuple(channels)
    if not selected_channels:
        raise ValueError("at least one channel is required")
    if not 0 <= window[0] < window[1]:
        raise ValueError(f"invalid trial window {window}")

    indices = _channel_indices(recording.channel_names, selected_channels)
    referenced = recording.samples - recording.samples.mean(axis=1, keepdims=True)
    selected = referenced[:, indices]
    start_offset = int(round(window[0] * recording.fs))
    stop_offset = int(round(window[1] * recording.fs))

    trials = []
    labels = []
    positions = []
    for position, marker_code in zip(recording.marker_positions, recording.marker_codes):
        onset = int(position) - 1
        start = onset + start_offset
        stop = onset + stop_offset
        if start < 0 or stop > selected.shape[0]:
            continue
        trials.append(selected[start:stop].T)
        labels.append(_marker_to_label(marker_code))
        positions.append(position)

    if not trials:
        raise ValueError("no valid trials were extracted")

    return BCICIVTrials(
        subject=recording.subject,
        trials=np.asarray(trials, dtype=np.float64),
        labels=np.asarray(labels, dtype=int),
        class_names=recording.class_names,
        channel_names=selected_channels,
        marker_positions=np.asarray(positions, dtype=int),
        fs=recording.fs,
        window=window,
    )


def extract_log_bandpower_features(
    recording: BCICIVRecording,
    channels: Iterable[str] = DEFAULT_MOTOR_CHANNELS,
    band: tuple[float, float] = DEFAULT_BAND,
    window: tuple[float, float] = DEFAULT_WINDOW,
) -> BCICIVFeatures:
    """Extract one log-variance bandpower feature per selected channel.

    The output is 8-D by default: C3, C4, Cz, FC3, FC4, CP3, CP4, CPz after
    common-average reference and 8-30 Hz band-pass filtering.
    """

    selected_channels = tuple(channels)
    if len(selected_channels) != 8:
        raise ValueError("this photonic baseline expects exactly 8 selected channels")
    if not 0 <= band[0] < band[1] < recording.fs / 2:
        raise ValueError(f"invalid band {band} for fs={recording.fs}")
    if not 0 <= window[0] < window[1]:
        raise ValueError(f"invalid trial window {window}")

    indices = _channel_indices(recording.channel_names, selected_channels)
    referenced = recording.samples - recording.samples.mean(axis=1, keepdims=True)
    selected = referenced[:, indices]
    filtered = _bandpass(selected, fs=recording.fs, band=band)

    start_offset = int(round(window[0] * recording.fs))
    stop_offset = int(round(window[1] * recording.fs))
    features = []
    labels = []
    positions = []
    for position, marker_code in zip(recording.marker_positions, recording.marker_codes):
        # Marker positions in the ASCII files are 1-based sample indices.
        onset = int(position) - 1
        start = onset + start_offset
        stop = onset + stop_offset
        if start < 0 or stop > filtered.shape[0]:
            continue
        segment = filtered[start:stop]
        features.append(np.log(np.var(segment, axis=0) + 1e-8))
        labels.append(_marker_to_label(marker_code))
        positions.append(position)

    if not features:
        raise ValueError("no valid trials were extracted")

    return BCICIVFeatures(
        subject=recording.subject,
        features=np.asarray(features, dtype=np.float64),
        labels=np.asarray(labels, dtype=int),
        class_names=recording.class_names,
        feature_names=tuple(f"{channel}_{band[0]:.0f}-{band[1]:.0f}Hz" for channel in selected_channels),
        marker_positions=np.asarray(positions, dtype=int),
        window=window,
        band=band,
    )


def _parse_nfo(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    parsed: dict[str, object] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "fs":
            parsed[key] = float(value)
        elif key in {"classes", "clab"}:
            parsed[key] = [item.strip() for item in value.split(",")]
        else:
            parsed[key] = value
    for required in ("fs", "classes", "clab"):
        if required not in parsed:
            raise ValueError(f"{path} is missing required field {required!r}")
    return parsed


def _channel_indices(all_channels: tuple[str, ...], selected_channels: tuple[str, ...]) -> list[int]:
    indices = []
    for channel in selected_channels:
        try:
            indices.append(all_channels.index(channel))
        except ValueError as exc:
            raise ValueError(f"channel {channel!r} not found in recording") from exc
    return indices


def _bandpass(samples: FloatArray, fs: float, band: tuple[float, float]) -> FloatArray:
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, samples, axis=0)


def _marker_to_label(marker_code: int) -> int:
    if marker_code == -1:
        return 0
    if marker_code == 1:
        return 1
    raise ValueError(f"unexpected marker code {marker_code}; expected +1 or -1")


def _map_local_to_global_labels(
    labels: IntArray,
    local_class_names: tuple[str, ...],
    global_class_to_index: dict[str, int],
) -> IntArray:
    mapped = np.empty_like(labels, dtype=int)
    for local_index, class_name in enumerate(local_class_names):
        if class_name not in global_class_to_index:
            raise ValueError(f"unsupported pooled class name {class_name!r}")
        mapped[labels == local_index] = global_class_to_index[class_name]
    return mapped


def _normalize_subject(subject: str) -> str:
    value = subject.lower().strip()
    if value.startswith("ds1"):
        value = value[-1]
    if value not in set("abcdefg"):
        raise ValueError("subject must be one of a, b, c, d, e, f, g")
    return value
