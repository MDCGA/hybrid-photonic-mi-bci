"""Cyton host utility for acquisition checks and experience-library management."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_photonic_mi_bci.host import (  # noqa: E402
    BrainFlowCytonConfig,
    BrainFlowCytonDevice,
    CytonHostController,
    ExperienceStore,
    SyntheticCytonDevice,
)
from hybrid_photonic_mi_bci.host.tk_app import run_app  # noqa: E402


def main() -> None:
    args = parse_args()
    if args.command == "gui":
        run_app(args.db)
        return
    store = ExperienceStore(args.db)
    if args.command == "init-db":
        store.initialize()
        group = store.get_active_group() or store.create_group(name="Cyton MI Default")
        print(f"initialized {store.db_path}")
        print(f"active group: {group.name} ({group.group_id})")
        return
    if args.command == "create-group":
        group = store.create_group(name=args.name, description=args.description)
        print(f"created group: {group.name} ({group.group_id})")
        return
    if args.command == "list-groups":
        groups = store.list_groups()
        if not groups:
            print("no groups")
            return
        for group in groups:
            marker = "*" if group.is_active else " "
            print(f"{marker} {group.group_id} | {group.name} | {group.device} | {','.join(group.channel_set)}")
        return
    if args.command == "stream":
        device = _build_device(args)
        controller = CytonHostController(store=store, device=device)
        controller.initialize_store()
        controller.ensure_default_group()
        controller.connect()
        controller.start_stream()
        try:
            deadline = time.time() + args.duration
            while time.time() < deadline:
                summary = controller.poll(max_samples=args.samples)
                rms = ", ".join(f"{name}:{value:.3f}" for name, value in summary.rms_by_channel.items())
                print(f"samples={summary.frame.n_samples} fs={summary.frame.sampling_rate:.1f} rms={rms}")
                time.sleep(args.interval)
        finally:
            controller.stop_stream()
            controller.disconnect()
        return
    raise ValueError(f"unsupported command {args.command}")


def _build_device(args: argparse.Namespace):
    if args.synthetic:
        return SyntheticCytonDevice()
    return BrainFlowCytonDevice(BrainFlowCytonConfig(serial_port=args.serial_port))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="artifacts/host/cyton_experience.sqlite")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db")
    create = subparsers.add_parser("create-group")
    create.add_argument("--name", required=True)
    create.add_argument("--description", default="")
    subparsers.add_parser("list-groups")
    stream = subparsers.add_parser("stream")
    stream.add_argument("--synthetic", action="store_true")
    stream.add_argument("--serial-port", default="COM3")
    stream.add_argument("--duration", type=float, default=2.0)
    stream.add_argument("--interval", type=float, default=0.5)
    stream.add_argument("--samples", type=int, default=250)
    subparsers.add_parser("gui")
    return parser.parse_args()


if __name__ == "__main__":
    main()
