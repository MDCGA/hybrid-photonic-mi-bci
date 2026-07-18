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

## Dataset protocols

For `BCICIV_1_asc`, the FastAPI demo intentionally matches
`examples/run_single_window_inference.py`: subjects `a` through `g` each
contribute their first 120 trials to pooled FBCSP/MLP/experience-library
training. The following six trials from every subject form a 42-window pooled
calibration set. For a selected subject, local trial 126 is therefore
evaluation index 0.

For `BNCI2014_004`, sessions 01T-02T train the subject-specific runtime. Sixteen
class-balanced windows from session 03T calibrate it, and the remaining session
03T windows are held out for evaluation.

At startup the service loads all seven BCICIV recordings and prepares subject
`a` so the default demonstration is immediately available. Other runtimes are
prepared on first use and cached in memory. Online inference calls
`run_one_online_forward()` from `examples/run_single_window_inference.py`.
Recursive SOS filtering stays on the full-precision SciPy signal backend;
CSP projection, feature standardization, MLP layers, and experience-head scan
continue through the installed photonic matrix backend.

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
