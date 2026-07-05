import tempfile
import unittest
from pathlib import Path

from hybrid_photonic_mi_bci.host import (
    CytonCommandBuilder,
    CytonHostController,
    ExperienceStore,
    SyntheticCytonDevice,
)


class CytonHostTest(unittest.TestCase):
    def test_synthetic_device_streams_cyton_like_frames(self) -> None:
        device = SyntheticCytonDevice(seed=1)
        device.connect()
        device.start_stream()

        frame = device.read_window(128)

        self.assertEqual(frame.samples.shape, (8, 128))
        self.assertEqual(frame.timestamps.shape, (128,))
        self.assertEqual(frame.sampling_rate, 250.0)
        self.assertEqual(frame.channel_names[0], "C3")
        device.stop_stream()
        device.disconnect()

    def test_experience_store_manages_groups_and_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ExperienceStore(Path(tmp) / "experience.sqlite")
            group = store.create_group(name="Lab Cyton", description="demo")
            entry = store.add_entry(
                group_id=group.group_id,
                subject_id="S01",
                session_id="2026-07-05",
                source="calibration",
                artifact_path=Path(tmp) / "entry.npz",
                metrics={"accuracy": 0.75},
                metadata={"operator": "test"},
            )

            self.assertTrue(store.get_group(group.group_id).is_active)
            self.assertEqual(len(store.list_groups()), 1)
            self.assertEqual(store.get_entry(entry.entry_id).metrics["accuracy"], 0.75)
            self.assertEqual(len(store.list_entries(group.group_id)), 1)

    def test_cyton_command_builder_validates_channel_settings(self) -> None:
        command = CytonCommandBuilder.channel_settings(
            channel=3,
            gain=24,
            input_type="normal",
            bias=True,
            srb2=True,
            srb1=False,
        )

        self.assertEqual(command, "x3060110X")
        with self.assertRaises(ValueError):
            CytonCommandBuilder.channel_settings(channel=9)

    def test_host_controller_poll_summary_uses_store_and_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = CytonHostController(
                store=ExperienceStore(Path(tmp) / "host.sqlite"),
                device=SyntheticCytonDevice(seed=2),
            )
            controller.initialize_store()
            group = controller.ensure_default_group()
            controller.connect()
            controller.start_stream()
            summary = controller.poll(64)

            self.assertTrue(group.group_id.startswith("group_"))
            self.assertEqual(summary.frame.samples.shape, (8, 64))
            self.assertEqual(len(summary.rms_by_channel), 8)
            self.assertGreater(len(controller.events), 0)
            controller.stop_stream()
            controller.disconnect()


if __name__ == "__main__":
    unittest.main()
