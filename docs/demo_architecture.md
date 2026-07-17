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

## Subject-personalized inference mode

For target subject `a`, subjects `b` through `g` provide the base training
trials. Their first 120 trials per subject fit FBCSP, the feature standardizer,
the compact MLP, and the candidate experience library. Subject `a` contributes
no samples to this base training stage.

From subject `a`'s trials after index 120, six class-balanced windows are used
only to select and weight Top-K experience heads and calibrate subject `a`'s
rejection threshold. Every remaining subject `a` replay window is held out for
single-window inference. The same leave-one-subject-out protocol is applied to
subjects `b` through `g`.

At startup the service loads all seven recordings and prepares subject `a` so
the default demonstration is immediately available. Other subject runtimes are
prepared on first use and cached in memory. Online inference still calls
`run_one_online_forward()` from `examples/run_single_window_inference.py`.

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
