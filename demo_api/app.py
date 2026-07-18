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

from demo_api.engine import DEFAULT_DATASET_KEY, LiveInferenceEngine, STAGES


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
    description="Real BCICIV/BNCI single-window inference for the external demo GUI.",
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
def health() -> dict[str, object]:
    datasets = ENGINE.dataset_options()
    LOGGER.info(
        "health_check mode=%s datasets=%s",
        "multi_dataset_subject_personalized_inference",
        [item["key"] for item in datasets],
    )
    return {
        "status": "ok",
        "mode": "multi_dataset_subject_personalized_inference",
        "datasets": datasets,
    }


@app.get("/api/demo")
def demo_metadata(
    request: Request,
    dataset: str = Query(default=DEFAULT_DATASET_KEY),
    subject: str | None = Query(default=None),
) -> dict[str, object]:
    client = request.client.host if request.client else "unknown"
    started_at = perf_counter()
    try:
        metadata = ENGINE.metadata(subject=subject, dataset=dataset)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOGGER.warning(
            "metadata_request_rejected client=%s dataset=%s subject=%s reason=%s",
            client,
            dataset,
            subject,
            exc,
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    LOGGER.info(
        "metadata_served client=%s dataset=%s subject=%s evaluation_count=%d calibration_windows=%d runtime_ready=%s elapsed_ms=%.3f",
        client,
        metadata["dataset_key"],
        metadata["subject"],
        metadata["evaluation_count"],
        metadata["calibration_windows"],
        metadata["runtime_ready"],
        (perf_counter() - started_at) * 1000.0,
    )
    return metadata


@app.get("/api/inference/stream")
async def inference_stream(
    request: Request,
    dataset: str = Query(default=DEFAULT_DATASET_KEY),
    subject: str | None = Query(default=None),
    evaluation_index: int = Query(default=0, ge=0),
) -> StreamingResponse:
    client = request.client.host if request.client else "unknown"
    try:
        window = ENGINE.window_info(
            subject=subject,
            evaluation_index=evaluation_index,
            dataset=dataset,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOGGER.warning(
            "inference_request_rejected client=%s dataset=%s subject=%s evaluation_index=%d reason=%s",
            client,
            dataset,
            subject,
            evaluation_index,
            exc,
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    LOGGER.info(
        "inference_requested client=%s dataset=%s subject=%s test_window=%d evaluation_index=%d subject_trial_index=%d true_label=%s runtime_ready=%s",
        client,
        window["dataset_key"],
        window["subject"],
        evaluation_index + 1,
        evaluation_index,
        window["subject_trial_index"],
        window["true_label"],
        ENGINE.runtime_ready(window["dataset_key"], window["subject"]),
    )

    async def event_stream():
        run_window = {key: value for key, value in window.items() if key != "signal"}
        LOGGER.info(
            "sse_event_sent event=run_started dataset=%s subject=%s evaluation_index=%d",
            window["dataset_key"],
            window["subject"],
            evaluation_index,
        )
        yield _sse(
            "run_started",
            {
                "type": "run_started",
                "run": {
                    **run_window,
                    "signal": None,
                    "mode": "multi_dataset_subject_personalized_inference",
                },
            },
        )
        window_stage = _stage_payload(STAGES[0], status="running")
        window_stage["signal"] = window["signal"]
        LOGGER.info(
            "sse_stage_started dataset=%s subject=%s stage=%s",
            window["dataset_key"],
            window["subject"],
            STAGES[0].stage_id,
        )
        yield _sse("stage_started", {"type": "stage_started", "stage": window_stage})
        await asyncio.sleep(0.25)
        LOGGER.info(
            "sse_stage_completed dataset=%s subject=%s stage=%s",
            window["dataset_key"],
            window["subject"],
            STAGES[0].stage_id,
        )
        yield _sse(
            "stage_completed",
            {
                "type": "stage_completed",
                "stage": _stage_payload(STAGES[0], status="completed"),
            },
        )

        fbcsp_stage = _stage_payload(STAGES[1], status="running")
        LOGGER.info(
            "sse_stage_started dataset=%s subject=%s stage=%s",
            window["dataset_key"],
            window["subject"],
            STAGES[1].stage_id,
        )
        yield _sse("stage_started", {"type": "stage_started", "stage": fbcsp_stage})
        try:
            result = await asyncio.to_thread(
                ENGINE.infer,
                window["subject"],
                evaluation_index,
                window["dataset_key"],
            )
        except Exception as exc:
            LOGGER.exception(
                "inference_failed dataset=%s subject=%s test_window=%d evaluation_index=%d",
                window["dataset_key"],
                window["subject"],
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
            "inference_completed dataset=%s subject=%s training_subjects=%s calibration_windows=%d test_window=%d evaluation_index=%d subject_trial_index=%d true_label=%s predicted_label=%s rejected=%s confidence=%.6f margin=%.6f online_ms=%.3f tiles=%d",
            result["dataset_key"],
            result["subject"],
            result["training_subjects"],
            result["calibration_windows"],
            evaluation_index + 1,
            evaluation_index,
            result["subject_trial_index"],
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
                LOGGER.info(
                    "sse_stage_started dataset=%s subject=%s stage=%s",
                    result["dataset_key"],
                    result["subject"],
                    stage.stage_id,
                )
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
                completed["duration_ms"] = result["online_timings_ms"].get(stage.timing_key, 0.0)
                completed["tile_count"] = result["online_tile_counts"].get(stage.timing_key, 0)
            if stage.stage_id == "photonic_scan":
                completed["top_k"] = len(result["selected_entries"])
                completed["tile_shape"] = [2, 8]
                completed["precision"] = result["precision"]
            LOGGER.info(
                "sse_stage_completed dataset=%s subject=%s stage=%s duration_ms=%s tile_count=%s",
                result["dataset_key"],
                result["subject"],
                stage.stage_id,
                completed.get("duration_ms"),
                completed.get("tile_count"),
            )
            yield _sse(
                "stage_completed",
                {"type": "stage_completed", "stage": completed},
            )
            await asyncio.sleep(0.28)

        result["notice"] = (
            f"{result['training_summary']}；{result['calibration_summary']}；"
            "当前测试窗口来自 held-out evaluation。"
        )
        LOGGER.info(
            "sse_event_sent event=run_completed dataset=%s subject=%s evaluation_index=%d",
            result["dataset_key"],
            result["subject"],
            evaluation_index,
        )
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
