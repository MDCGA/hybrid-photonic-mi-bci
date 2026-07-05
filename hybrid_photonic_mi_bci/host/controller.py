"""Application controller tying acquisition, experience storage, and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .acquisition import AcquisitionDevice, SyntheticCytonDevice
from .experience_store import ExperienceGroupRecord, ExperienceStore
from .models import HostEvent, StreamFrame


@dataclass(frozen=True)
class PollSummary:
    frame: StreamFrame
    rms_by_channel: dict[str, float]


class CytonHostController:
    """Thin controller suitable for CLI, Tkinter, or a future Qt UI."""

    def __init__(
        self,
        store: ExperienceStore | None = None,
        device: AcquisitionDevice | None = None,
    ):
        self.store = store or ExperienceStore()
        self.device = device or SyntheticCytonDevice()
        self.events: list[HostEvent] = []
        self.connected = False
        self.streaming = False

    @classmethod
    def synthetic(cls, db_path: str | Path = "artifacts/host/cyton_experience.sqlite") -> "CytonHostController":
        return cls(store=ExperienceStore(db_path), device=SyntheticCytonDevice())

    def initialize_store(self) -> None:
        self.store.initialize()
        self._event("store", "experience store initialized", {"db": str(self.store.db_path)})

    def ensure_default_group(self) -> ExperienceGroupRecord:
        self.store.initialize()
        active = self.store.get_active_group()
        if active is not None:
            return active
        group = self.store.create_group(
            name="Cyton MI Default",
            description="Default OpenBCI Cyton motor-imagery experience group.",
        )
        self._event("library", f"created default group {group.name}", {"group_id": group.group_id})
        return group

    def connect(self) -> None:
        self.device.connect()
        self.connected = True
        self._event("device", "device connected")

    def disconnect(self) -> None:
        if self.streaming:
            self.stop_stream()
        self.device.disconnect()
        self.connected = False
        self._event("device", "device disconnected")

    def start_stream(self) -> None:
        if not self.connected:
            self.connect()
        self.device.start_stream()
        self.streaming = True
        self._event("stream", "stream started")

    def stop_stream(self) -> None:
        self.device.stop_stream()
        self.streaming = False
        self._event("stream", "stream stopped")

    def send_board_command(self, command: str) -> str:
        response = self.device.send_command(command)
        self._event("command", response, {"command": command})
        return response

    def poll(self, max_samples: int = 250) -> PollSummary:
        frame = self.device.read_window(max_samples)
        rms = {}
        if frame.samples.size:
            values = np.sqrt(np.mean(frame.samples**2, axis=1))
            rms = {
                channel: float(value)
                for channel, value in zip(frame.channel_names, values)
            }
        self._event("poll", f"read {frame.n_samples} samples", {"rms": rms})
        return PollSummary(frame=frame, rms_by_channel=rms)

    def _event(self, event_type: str, message: str, payload: dict[str, object] | None = None) -> None:
        self.events.append(HostEvent(event_type=event_type, message=message, payload=payload or {}))
