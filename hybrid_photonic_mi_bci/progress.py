"""Small progress and timing recorder for long-running workflows."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
from typing import Iterator


@dataclass
class ProgressEvent:
    step: str
    status: str
    started_at: str
    ended_at: str | None
    elapsed_seconds: float | None
    index: int | None = None
    total: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed_seconds": self.elapsed_seconds,
            "index": self.index,
            "total": self.total,
        }


class ProgressLogger:
    """Print progress updates and persist per-step elapsed time."""

    def __init__(
        self,
        run_name: str,
        output_path: str | Path | None = None,
        *,
        print_updates: bool = True,
    ) -> None:
        self.run_name = str(run_name)
        self.output_path = Path(output_path) if output_path is not None else None
        self.print_updates = bool(print_updates)
        self.started_at = _now()
        self._start_counter = perf_counter()
        self.events: list[ProgressEvent] = []

    @contextmanager
    def step(
        self,
        name: str,
        *,
        index: int | None = None,
        total: int | None = None,
    ) -> Iterator[None]:
        prefix = _format_prefix(index, total)
        if self.print_updates:
            print(f"[progress] {self.run_name} {prefix}start: {name}")
        event = ProgressEvent(
            step=name,
            status="running",
            started_at=_now(),
            ended_at=None,
            elapsed_seconds=None,
            index=index,
            total=total,
        )
        step_start = perf_counter()
        try:
            yield
        except Exception:
            elapsed = perf_counter() - step_start
            event.status = "failed"
            event.ended_at = _now()
            event.elapsed_seconds = elapsed
            self.events.append(event)
            self.write()
            if self.print_updates:
                print(f"[progress] {self.run_name} {prefix}failed: {name} ({elapsed:.2f}s)")
            raise
        else:
            elapsed = perf_counter() - step_start
            event.status = "completed"
            event.ended_at = _now()
            event.elapsed_seconds = elapsed
            self.events.append(event)
            self.write()
            if self.print_updates:
                print(f"[progress] {self.run_name} {prefix}done: {name} ({elapsed:.2f}s)")

    def write(self) -> None:
        if self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        total_elapsed = perf_counter() - self._start_counter
        payload = {
            "run_name": self.run_name,
            "started_at": self.started_at,
            "updated_at": _now(),
            "elapsed_seconds": total_elapsed,
            "events": [event.to_dict() for event in self.events],
        }
        self.output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _format_prefix(index: int | None, total: int | None) -> str:
    if index is None or total is None:
        return ""
        return f"({index}/{total}) "


class ConsoleProgressBar:
    """Small dependency-free terminal progress bar."""

    def __init__(
        self,
        label: str,
        total: int,
        *,
        width: int = 28,
        enabled: bool = True,
        min_interval_seconds: float = 0.10,
    ) -> None:
        self.label = str(label)
        self.total = max(0, int(total))
        self.width = max(4, int(width))
        self.enabled = bool(enabled)
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._started = perf_counter()
        self._last_update = 0.0
        self._last_length = 0
        self._printed = False

    def update(self, current: int, *, suffix: str = "") -> None:
        if not self.enabled or self.total <= 0:
            return
        current = min(max(0, int(current)), self.total)
        now = perf_counter()
        if current < self.total and now - self._last_update < self.min_interval_seconds:
            return
        fraction = current / self.total
        filled = int(round(self.width * fraction))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self._started
        rate = current / elapsed if elapsed > 0.0 else 0.0
        eta = (self.total - current) / rate if rate > 0.0 else 0.0
        message = (
            f"\r[progress] {self.label} [{bar}] "
            f"{current}/{self.total} {fraction * 100.0:5.1f}% "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
        )
        if suffix:
            message += f" {suffix}"
        padded = message.ljust(self._last_length)
        self._last_length = len(message)
        print(padded, end="", flush=True)
        self._last_update = now
        self._printed = True

    def close(self) -> None:
        if self.enabled and self._printed:
            print()
