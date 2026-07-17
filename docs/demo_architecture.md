# Single-window inference demo

The FastAPI backend code runs directly inside the existing Docker container.
No image build is required. The browser GUI runs outside the container and
consumes HTTP metadata plus an SSE inference event stream through an exposed
port.

## Flow

1. Load one real 3.0-second EEG evaluation window.
2. Run filter-bank and CSP spatial projection.
3. Select and standardize 32 FBCSP features.
4. Produce a 32-dimensional compact-MLP embedding.
5. Scan Top-K experience heads on 2x8 photonic tiles.
6. Fuse probabilities, apply rejection, and display the command.

## Live inference mode

The service directly reuses `prepare_runtime()` and `run_one_online_forward()`
from `examples/run_single_window_inference.py`. At FastAPI startup it loads the
real BCICIV_1_asc trials, fits FBCSP, trains the compact MLP, builds the
experience library, calibrates the runtime, and keeps the prepared runtime in
memory. Each inference request selects one held-out evaluation window and runs
the complete online forward path once.

The EEG waveform, class probabilities, rejection decision, timings, selected
experience entries, adaptive-precision telemetry, and tile counts returned to
the GUI are produced by that real request. Startup can take significant time
because it intentionally follows the example's offline preparation exactly.

## Start

Inside the existing backend container, from the repository root:

```powershell
python scripts/download_datasets.py --dataset bciciv-1-asc
python -m pip install -e ".[demo]"
python -m uvicorn demo_api.app:app --host 0.0.0.0 --port 8000
```

Expose or publish container port `8000` to the host. Port publishing is set
when the container is created; for example, the equivalent Docker option is
`-p 8000:8000`. No project image or Compose build file is needed. FastAPI's
interactive API documentation is available at `/docs`.

On the host:

```powershell
cd hybrid-photonic-mi-bci-gui
python -m http.server 5173
```

Open `http://localhost:5173`. Override the API address with a query string such
as `?api=http://192.168.1.10:8000`.
