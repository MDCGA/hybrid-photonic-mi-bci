"""Loader for BCI Competition IV 2b / BNCI2014_004 GDF files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]

DEFAULT_CHANNELS = ("EEG:C3", "EEG:Cz", "EEG:C4")
DEFAULT_WINDOW = (0.5, 3.5)
DEFAULT_SUBJECTS = tuple(range(1, 10))
CLASS_NAMES = ("left", "right")


@dataclass(frozen=True)
class BNCI2014_004Session:
    subject: int
    session: int
    trials: FloatArray
    labels: IntArray
    class_names: tuple[str, ...]
    channel_names: tuple[str, ...]
    fs: float
    window: tuple[float, float]


def load_session(
    data_dir: str | Path,
    subject: int,
    session: int,
    channels: Iterable[str] = DEFAULT_CHANNELS,
    window: tuple[float, float] = DEFAULT_WINDOW,
) -> BNCI2014_004Session:
    """Load one labeled 2b training session.

    Only ``T`` files are used because their left/right labels are included as
    GDF annotations. The usual protocol has sessions 1-3 labeled and 4-5
    evaluation-only.
    """

    if session not in {1, 2, 3}:
        raise ValueError("this loader uses labeled T sessions 1, 2, or 3")
    path = Path(data_dir) / "gdf" / f"B{subject:02d}{session:02d}T.gdf"
    if not path.exists():
        raise FileNotFoundError(path)
    return load_gdf_trials(path, subject=subject, session=session, channels=channels, window=window)


def load_subject_history_and_target(
    data_dir: str | Path,
    subject: int,
    history_sessions: tuple[int, ...] = (1, 2),
    target_session: int = 3,
    channels: Iterable[str] = DEFAULT_CHANNELS,
    window: tuple[float, float] = DEFAULT_WINDOW,
) -> tuple[BNCI2014_004Session, BNCI2014_004Session]:
    """Return concatenated history sessions and a target session."""

    history = [
        load_session(data_dir, subject, session, channels=channels, window=window)
        for session in history_sessions
    ]
    target = load_session(data_dir, subject, target_session, channels=channels, window=window)
    trials = np.concatenate([item.trials for item in history], axis=0)
    labels = np.concatenate([item.labels for item in history], axis=0)
    return (
        BNCI2014_004Session(
            subject=subject,
            session=0,
            trials=trials,
            labels=labels,
            class_names=CLASS_NAMES,
            channel_names=history[0].channel_names,
            fs=history[0].fs,
            window=window,
        ),
        target,
    )


def load_gdf_trials(
    path: str | Path,
    subject: int,
    session: int,
    channels: Iterable[str] = DEFAULT_CHANNELS,
    window: tuple[float, float] = DEFAULT_WINDOW,
) -> BNCI2014_004Session:
    """Extract left/right cue-locked epochs from one 2b GDF file."""

    import mne

    selected_channels = tuple(channels)
    raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
    raw.pick(list(selected_channels))
    data = raw.get_data().astype(np.float64)
    data = np.nan_to_num(data, copy=False)
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    code_to_event = {str(key): value for key, value in event_id.items()}
    missing = {"769", "770"} - set(code_to_event)
    if missing:
        raise ValueError(f"{path} is missing class cue annotations {sorted(missing)}")
    fs = float(raw.info["sfreq"])
    start_offset = int(round(window[0] * fs))
    stop_offset = int(round(window[1] * fs))
    trials = []
    labels = []
    for sample, _previous, event_code in events:
        if event_code == code_to_event["769"]:
            label = 0
        elif event_code == code_to_event["770"]:
            label = 1
        else:
            continue
        start = int(sample) + start_offset
        stop = int(sample) + stop_offset
        if start < 0 or stop > data.shape[1]:
            continue
        trials.append(data[:, start:stop])
        labels.append(label)
    if not trials:
        raise ValueError(f"no labeled trials extracted from {path}")
    return BNCI2014_004Session(
        subject=subject,
        session=session,
        trials=np.asarray(trials, dtype=np.float64),
        labels=np.asarray(labels, dtype=int),
        class_names=CLASS_NAMES,
        channel_names=selected_channels,
        fs=fs,
        window=window,
    )


def calibration_eval_split(
    labels: IntArray,
    trials_per_class: int,
    n_classes: int = 2,
) -> tuple[IntArray, IntArray]:
    """Select the first ``trials_per_class`` per class for calibration."""

    labels_arr = np.asarray(labels, dtype=int)
    calibration = []
    for class_index in range(n_classes):
        indices = np.flatnonzero(labels_arr == class_index)
        if len(indices) <= trials_per_class:
            raise ValueError("calibration split leaves no evaluation samples")
        calibration.extend(indices[:trials_per_class].tolist())
    calibration_idx = np.asarray(sorted(calibration), dtype=int)
    mask = np.ones(len(labels_arr), dtype=bool)
    mask[calibration_idx] = False
    return calibration_idx, np.flatnonzero(mask).astype(int)
