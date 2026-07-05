"""Acquisition adapters for OpenBCI Cyton and offline development."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .models import StreamFrame


DEFAULT_CYTON_CHANNELS = ("C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4", "CPz")


class AcquisitionDevice(ABC):
    """Minimal board-control contract used by the host controller."""

    @abstractmethod
    def connect(self) -> None:
        """Prepare the device session."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release the device session."""

    @abstractmethod
    def start_stream(self) -> None:
        """Start acquisition."""

    @abstractmethod
    def stop_stream(self) -> None:
        """Stop acquisition."""

    @abstractmethod
    def read_window(self, max_samples: int) -> StreamFrame:
        """Read up to ``max_samples`` without blocking indefinitely."""

    @abstractmethod
    def send_command(self, command: str) -> str:
        """Send a board-specific command and return a short status string."""


@dataclass(frozen=True)
class BrainFlowCytonConfig:
    """Connection settings for a Cyton board through BrainFlow."""

    serial_port: str
    board_id: int | None = None
    timeout: int = 15
    channel_names: tuple[str, ...] = DEFAULT_CYTON_CHANNELS


class BrainFlowCytonDevice(AcquisitionDevice):
    """OpenBCI Cyton adapter backed by BrainFlow.

    BrainFlow is optional so the repository can be tested without hardware.
    Install it with ``python -m pip install -e ".[cyton]"``.
    """

    def __init__(self, config: BrainFlowCytonConfig):
        self.config = config
        self._board: Any | None = None
        self._board_id: int | None = None
        self._eeg_channels: list[int] | None = None
        self._timestamp_channel: int | None = None
        self._sampling_rate: float | None = None
        self._streaming = False

    def connect(self) -> None:
        try:
            from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
        except ImportError as exc:
            raise RuntimeError(
                "BrainFlow is not installed. Install with: "
                "python -m pip install -e \".[cyton]\""
            ) from exc

        board_id = self.config.board_id
        if board_id is None:
            board_id = int(BoardIds.CYTON_BOARD.value)
        params = BrainFlowInputParams()
        params.serial_port = self.config.serial_port
        params.timeout = self.config.timeout
        board = BoardShim(board_id, params)
        board.prepare_session()
        self._board = board
        self._board_id = board_id
        self._eeg_channels = list(BoardShim.get_eeg_channels(board_id))[: len(self.config.channel_names)]
        self._timestamp_channel = int(BoardShim.get_timestamp_channel(board_id))
        self._sampling_rate = float(BoardShim.get_sampling_rate(board_id))

    def disconnect(self) -> None:
        if self._board is None:
            return
        if self._streaming:
            self.stop_stream()
        self._board.release_session()
        self._board = None
        self._board_id = None

    def start_stream(self) -> None:
        board = self._require_board()
        if not self._streaming:
            board.start_stream()
            self._streaming = True

    def stop_stream(self) -> None:
        board = self._require_board()
        if self._streaming:
            board.stop_stream()
            self._streaming = False

    def read_window(self, max_samples: int) -> StreamFrame:
        board = self._require_board()
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        raw = board.get_current_board_data(max_samples)
        eeg_channels = self._eeg_channels or []
        timestamps = raw[self._timestamp_channel] if self._timestamp_channel is not None else np.array([])
        return StreamFrame(
            samples=np.asarray(raw[eeg_channels], dtype=np.float64),
            timestamps=np.asarray(timestamps, dtype=np.float64),
            channel_names=self.config.channel_names[: len(eeg_channels)],
            sampling_rate=float(self._sampling_rate or 0.0),
        )

    def send_command(self, command: str) -> str:
        board = self._require_board()
        board.config_board(command)
        return f"sent {command!r}"

    def _require_board(self):
        if self._board is None:
            raise RuntimeError("device is not connected")
        return self._board


class SyntheticCytonDevice(AcquisitionDevice):
    """Deterministic Cyton-like signal source for UI and pipeline development."""

    def __init__(
        self,
        sampling_rate: float = 250.0,
        channel_names: tuple[str, ...] = DEFAULT_CYTON_CHANNELS,
        seed: int = 13,
    ):
        self.sampling_rate = float(sampling_rate)
        self.channel_names = channel_names
        self._rng = np.random.default_rng(seed)
        self._connected = False
        self._streaming = False
        self._sample_index = 0

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._streaming = False
        self._connected = False

    def start_stream(self) -> None:
        self._require_connected()
        self._streaming = True

    def stop_stream(self) -> None:
        self._require_connected()
        self._streaming = False

    def read_window(self, max_samples: int) -> StreamFrame:
        self._require_connected()
        if not self._streaming:
            return StreamFrame(
                samples=np.empty((len(self.channel_names), 0), dtype=np.float64),
                timestamps=np.empty(0, dtype=np.float64),
                channel_names=self.channel_names,
                sampling_rate=self.sampling_rate,
            )
        n_samples = max(1, int(max_samples))
        indices = self._sample_index + np.arange(n_samples)
        t = indices / self.sampling_rate
        self._sample_index += n_samples
        samples = []
        for channel_index, _name in enumerate(self.channel_names):
            mu = np.sin(2 * np.pi * (10.0 + 0.2 * channel_index) * t)
            beta = 0.45 * np.sin(2 * np.pi * (20.0 + 0.15 * channel_index) * t)
            drift = 0.15 * np.sin(2 * np.pi * 0.35 * t + channel_index)
            noise = self._rng.normal(scale=0.08, size=n_samples)
            samples.append(mu + beta + drift + noise)
        return StreamFrame(
            samples=np.asarray(samples, dtype=np.float64),
            timestamps=time.time() + t,
            channel_names=self.channel_names,
            sampling_rate=self.sampling_rate,
        )

    def send_command(self, command: str) -> str:
        self._require_connected()
        return f"synthetic accepted {command!r}"

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("device is not connected")


class CytonCommandBuilder:
    """Small helper for Cyton ASCII configuration commands.

    Streaming is controlled through BrainFlow methods in the host app. This
    builder covers common low-level commands that are useful from an operator UI.
    """

    GAIN_CODES = {
        1: "0",
        2: "1",
        4: "2",
        6: "3",
        8: "4",
        12: "5",
        24: "6",
    }
    INPUT_CODES = {
        "normal": "0",
        "shorted": "1",
        "bias_meas": "2",
        "mvdd": "3",
        "temp": "4",
        "testsig": "5",
        "bias_drp": "6",
        "bias_drn": "7",
    }

    @staticmethod
    def default_channel_settings() -> str:
        return "d"

    @staticmethod
    def query_registers() -> str:
        return "?"

    @classmethod
    def channel_settings(
        cls,
        channel: int,
        power_down: bool = False,
        gain: int = 24,
        input_type: str = "normal",
        bias: bool = True,
        srb2: bool = True,
        srb1: bool = False,
    ) -> str:
        if not 1 <= int(channel) <= 8:
            raise ValueError("Cyton channel must be in 1..8")
        if gain not in cls.GAIN_CODES:
            raise ValueError(f"unsupported gain {gain}; expected one of {sorted(cls.GAIN_CODES)}")
        if input_type not in cls.INPUT_CODES:
            raise ValueError(f"unsupported input_type {input_type!r}")
        return "".join(
            [
                "x",
                str(int(channel)),
                "1" if power_down else "0",
                cls.GAIN_CODES[gain],
                cls.INPUT_CODES[input_type],
                "1" if bias else "0",
                "1" if srb2 else "0",
                "1" if srb1 else "0",
                "X",
            ]
        )
