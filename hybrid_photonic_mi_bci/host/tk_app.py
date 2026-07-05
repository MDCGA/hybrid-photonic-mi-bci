"""Small Tkinter host UI for Cyton control and experience-library management."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from .acquisition import BrainFlowCytonConfig, BrainFlowCytonDevice, CytonCommandBuilder, SyntheticCytonDevice
from .controller import CytonHostController
from .experience_store import ExperienceStore


class CytonHostApp(tk.Tk):
    """Minimal upper-computer UI that can grow into the full operator console."""

    def __init__(self, db_path: str | Path = "artifacts/host/cyton_experience.sqlite"):
        super().__init__()
        self.title("Hybrid Photonic MI-BCI Cyton Host")
        self.geometry("980x680")
        self.db_path = Path(db_path)
        self.controller = CytonHostController.synthetic(self.db_path)
        self.mode_var = tk.StringVar(value="synthetic")
        self.serial_var = tk.StringVar(value="COM3")
        self.group_name_var = tk.StringVar(value="Cyton MI Default")
        self.status_var = tk.StringVar(value="Idle")
        self._build()
        self._initialize_store()

    def _build(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        top = ttk.LabelFrame(root, text="Cyton Acquisition")
        top.pack(fill=tk.X)
        ttk.Label(top, text="Mode").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(top, textvariable=self.mode_var, values=("synthetic", "brainflow"), width=12).grid(
            row=0,
            column=1,
            padx=4,
            pady=4,
        )
        ttk.Label(top, text="Serial Port").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        ttk.Entry(top, textvariable=self.serial_var, width=16).grid(row=0, column=3, padx=4, pady=4)
        ttk.Button(top, text="Connect", command=self._connect).grid(row=0, column=4, padx=4, pady=4)
        ttk.Button(top, text="Start", command=self._start).grid(row=0, column=5, padx=4, pady=4)
        ttk.Button(top, text="Stop", command=self._stop).grid(row=0, column=6, padx=4, pady=4)
        ttk.Button(top, text="Poll", command=self._poll).grid(row=0, column=7, padx=4, pady=4)
        ttk.Label(top, textvariable=self.status_var).grid(row=1, column=0, columnspan=8, sticky="w", padx=4)

        cmd = ttk.LabelFrame(root, text="Board Commands")
        cmd.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(cmd, text="Default Channels", command=self._send_default_channels).pack(side=tk.LEFT, padx=4, pady=6)
        ttk.Button(cmd, text="Query Registers", command=self._query_registers).pack(side=tk.LEFT, padx=4, pady=6)

        library = ttk.LabelFrame(root, text="Experience Library Groups")
        library.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        form = ttk.Frame(library)
        form.pack(fill=tk.X, pady=4)
        ttk.Label(form, text="Group Name").pack(side=tk.LEFT, padx=4)
        ttk.Entry(form, textvariable=self.group_name_var, width=32).pack(side=tk.LEFT, padx=4)
        ttk.Button(form, text="Create Group", command=self._create_group).pack(side=tk.LEFT, padx=4)
        ttk.Button(form, text="Refresh", command=self._refresh_groups).pack(side=tk.LEFT, padx=4)

        self.group_tree = ttk.Treeview(
            library,
            columns=("active", "device", "channels", "created"),
            show="tree headings",
            height=8,
        )
        self.group_tree.heading("#0", text="Name")
        self.group_tree.heading("active", text="Active")
        self.group_tree.heading("device", text="Device")
        self.group_tree.heading("channels", text="Channels")
        self.group_tree.heading("created", text="Created")
        self.group_tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        log_frame = ttk.LabelFrame(root, text="Host Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.log = tk.Text(log_frame, height=10)
        self.log.pack(fill=tk.BOTH, expand=True)

    def _initialize_store(self) -> None:
        self.controller.initialize_store()
        self.controller.ensure_default_group()
        self._refresh_groups()
        self._write_log("Experience store ready.")

    def _connect(self) -> None:
        try:
            mode = self.mode_var.get()
            store = ExperienceStore(self.db_path)
            if mode == "brainflow":
                device = BrainFlowCytonDevice(BrainFlowCytonConfig(serial_port=self.serial_var.get()))
            else:
                device = SyntheticCytonDevice()
            self.controller = CytonHostController(store=store, device=device)
            self.controller.connect()
            self.status_var.set(f"Connected: {mode}")
            self._write_log(f"Connected using {mode}.")
        except Exception as exc:
            messagebox.showerror("Connect failed", str(exc))

    def _start(self) -> None:
        try:
            self.controller.start_stream()
            self.status_var.set("Streaming")
            self._write_log("Stream started.")
        except Exception as exc:
            messagebox.showerror("Start failed", str(exc))

    def _stop(self) -> None:
        try:
            self.controller.stop_stream()
            self.status_var.set("Stopped")
            self._write_log("Stream stopped.")
        except Exception as exc:
            messagebox.showerror("Stop failed", str(exc))

    def _poll(self) -> None:
        try:
            summary = self.controller.poll(max_samples=250)
            rms = ", ".join(f"{key}:{value:.3f}" for key, value in summary.rms_by_channel.items())
            self._write_log(f"Read {summary.frame.n_samples} samples. RMS {rms}")
        except Exception as exc:
            messagebox.showerror("Poll failed", str(exc))

    def _send_default_channels(self) -> None:
        self._send_command(CytonCommandBuilder.default_channel_settings())

    def _query_registers(self) -> None:
        self._send_command(CytonCommandBuilder.query_registers())

    def _send_command(self, command: str) -> None:
        try:
            response = self.controller.send_board_command(command)
            self._write_log(response)
        except Exception as exc:
            messagebox.showerror("Command failed", str(exc))

    def _create_group(self) -> None:
        name = self.group_name_var.get().strip()
        if not name:
            messagebox.showwarning("Missing name", "Group name is required.")
            return
        group = self.controller.store.create_group(name=name, description="Created from host UI.")
        self._write_log(f"Created group {group.name} ({group.group_id}).")
        self._refresh_groups()

    def _refresh_groups(self) -> None:
        for item in self.group_tree.get_children():
            self.group_tree.delete(item)
        for group in self.controller.store.list_groups():
            self.group_tree.insert(
                "",
                tk.END,
                iid=group.group_id,
                text=group.name,
                values=(
                    "yes" if group.is_active else "",
                    group.device,
                    ",".join(group.channel_set),
                    group.created_at,
                ),
            )

    def _write_log(self, message: str) -> None:
        self.log.insert(tk.END, f"{message}\n")
        self.log.see(tk.END)


def run_app(db_path: str | Path = "artifacts/host/cyton_experience.sqlite") -> None:
    app = CytonHostApp(db_path=db_path)
    app.mainloop()
