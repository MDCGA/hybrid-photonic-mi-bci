"""FastAPI entry point for the external single-window demo GUI."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
from time import perf_counter

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from demo_api.engine import LiveInferenceEngine, STAGES


ENGINE = LiveInferenceEngine()
LOGGER = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    started_at = perf_counter()
    try:
        await asyncio.to_thread(ENGINE.prepare)
    except Exception:
        LOGGER.exception("application_startup_failed")
        raise
    LOGGER.info(
        "application_ready startup_ms=%.3f",
        (perf_counter() - started_at) * 1000.0,
    )
    try:
        yield
    finally:
        LOGGER.info("application_shutdown")

app = FastAPI(
    title="Hybrid Photonic MI-BCI Demo API",
    version="0.1.0",
    description="Real BCICIV single-window inference for the external demo GUI.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "live_inference"}


@app.get("/api/demo")
def demo_metadata() -> dict[str, object]:
    return ENGINE.metadata()


@app.get("/api/inference/stream")
async def inference_stream(
    request: Request,
    evaluation_index: int = Query(default=0, ge=0),
) -> StreamingResponse:
    try:
        window = ENGINE.window_info(evaluation_index)
    except ValueError as exc:
        LOGGER.warning(
            "inference_request_rejected evaluation_index=%d reason=%s",
            evaluation_index,
            exc,
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    client = request.client.host if request.client else "unknown"
    LOGGER.info(
        "inference_requested client=%s test_window=%d evaluation_index=%d absolute_trial_index=%d true_label=%s",
        client,
        evaluation_index + 1,
        evaluation_index,
        window["absolute_trial_index"],
        window["true_label"],
    )

    async def event_stream():
        yield _sse(
            "run_started",
            {
                "type": "run_started",
                "run": {
                    **window,
                    "signal": None,
                    "dataset": "BCICIV_1_asc",
                    "mode": "live_inference",
                },
            },
        )
        window_stage = _stage_payload(STAGES[0], status="running")
        window_stage["signal"] = window["signal"]
        yield _sse("stage_started", {"type": "stage_started", "stage": window_stage})
        await asyncio.sleep(0.25)
        yield _sse(
            "stage_completed",
            {
                "type": "stage_completed",
                "stage": _stage_payload(STAGES[0], status="completed"),
            },
        )

        fbcsp_stage = _stage_payload(STAGES[1], status="running")
        yield _sse("stage_started", {"type": "stage_started", "stage": fbcsp_stage})
        try:
            result = await asyncio.to_thread(ENGINE.infer, evaluation_index)
        except Exception as exc:
            LOGGER.exception(
                "inference_failed test_window=%d evaluation_index=%d",
                evaluation_index + 1,
                evaluation_index,
            )
            yield _sse(
                "run_failed",
                {
                    "type": "run_failed",
                    "error": "推理执行失败",
                    "detail": str(exc),
                },
            )
            return

        LOGGER.info(
            "inference_completed test_window=%d evaluation_index=%d absolute_trial_index=%d true_label=%s predicted_label=%s rejected=%s confidence=%.6f margin=%.6f online_ms=%.3f tiles=%d",
            evaluation_index + 1,
            evaluation_index,
            result["absolute_trial_index"],
            result["true_label"],
            result["predicted_label"],
            result["rejected"],
            result["confidence"],
            result["margin"],
            result["online_total_ms"],
            result["tile_count"],
        )

        for stage in STAGES[1:]:
            if stage.stage_id != "fbcsp":
                yield _sse(
                    "stage_started",
                    {
                        "type": "stage_started",
                        "stage": _stage_payload(stage, status="running"),
                    },
                )
                await asyncio.sleep(0.18)
            completed = _stage_payload(stage, status="completed")
            if stage.timing_key:
                completed["duration_ms"] = result["online_timings_ms"][stage.timing_key]
                completed["tile_count"] = result["online_tile_counts"][stage.timing_key]
            if stage.stage_id == "photonic_scan":
                completed["top_k"] = len(result["selected_entries"])
                completed["tile_shape"] = [2, 8]
                completed["precision"] = result["precision"]
            yield _sse(
                "stage_completed",
                {"type": "stage_completed", "stage": completed},
            )
            await asyncio.sleep(0.28)

        result["notice"] = "EEG window, probabilities, decision, timing, and tile counts are produced by the real single-window interface."
        yield _sse("run_completed", {"type": "run_completed", "result": result})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _stage_payload(stage, *, status: str) -> dict[str, object]:
    return {
        "id": stage.stage_id,
        "title": stage.title,
        "description": stage.description,
        "status": status,
    }


def _sse(event_type: str, payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_type}\ndata: {serialized}\n\n"
