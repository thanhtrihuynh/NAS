from pathlib import Path
import os
import time

import importlib.util
import sys

import pydantic

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from dashboard_full.services.candidate_service import CandidateService
from dashboard_full.services.dataset_service import DatasetService
from dashboard_full.services.inference_service import InferenceService
from dashboard_full.services.model_service import ModelService
from dashboard_full.services.nas_service import NasService
from dashboard_full.services.pynq_service import PynqService
from dashboard_full.services.fpga_resource_service import FpgaResourceService
from dashboard_full.services.batch_test_service import BatchTestService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="ECG NAS FPGA Dashboard",
    version="5.19.0",
)

app.mount(
    "/static",
    StaticFiles(directory=DASHBOARD_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(
    directory=DASHBOARD_DIR / "templates"
)

dataset_service = DatasetService(PROJECT_ROOT)
model_service = ModelService(PROJECT_ROOT)
inference_service = InferenceService(PROJECT_ROOT)
candidate_service = CandidateService(PROJECT_ROOT)
nas_service = NasService(PROJECT_ROOT)
fpga_resource_service = FpgaResourceService(PROJECT_ROOT)

PYNQ_HOST = os.environ.get(
    "ECG_PYNQ_HOST",
    "192.168.2.99",
)

PYNQ_PORT = int(
    os.environ.get(
        "ECG_PYNQ_PORT",
        "9000",
    )
)

PYNQ_MODE = os.environ.get(
    "ECG_PYNQ_MODE",
    "hybrid",
)


batch_test_service = BatchTestService(
    project_root=PROJECT_ROOT,
    dataset_service=dataset_service,
    inference_service=inference_service,
    pynq_service=PynqService,
    pynq_host=PYNQ_HOST,
    pynq_port=PYNQ_PORT,
    pynq_mode=PYNQ_MODE,
)


class BeatRequest(BaseModel):
    beat_token: str | None = None
    sample_index: int | None = None


class SendRequest(BaseModel):
    beat_token: str


class FullTestRequest(BaseModel):
    mode: str = "local"


class NasSearchRequest(BaseModel):
    kernels: list[int]
    channels: list[int]
    trials: int = 20
    objective: str = "balanced"
    epochs_per_candidate: int = 5


class CandidateSelectionRequest(BaseModel):
    candidate_id: int | str
    job_id: str | None = None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


@app.get("/api/status")
def status():
    dataset = dataset_service.dataset_status()

    return {
        "server": "online",
        "project_root": str(PROJECT_ROOT),
        "test_csv": dataset["test_csv"],
        "records_dir": dataset["datasets_dir"],
        "records_found": dataset["records_found"],
        "test_samples": dataset["test_rows"],
        "labels": dataset_service.labels(),
    }



@app.post("/api/dataset/refresh")
def refresh_dataset():
    dataset_service.refresh()

    return {
        "ok": True,
        **dataset_service.dataset_status(),
    }


@app.get("/api/debug/dataset")
def debug_dataset():
    status = dataset_service.dataset_status()

    status["records"] = [
        {
            "record_id": item["record_id"],
            "filename": item["filename"],
            "record_base": item["record_base"],
            "has_atr": item["has_atr"],
            "channels": item["channels"],
        }
        for item in dataset_service.records()
    ]

    return status


@app.get("/api/records")
def records(
    label: str | None = None,
    query: str | None = None,
):
    return {
        "records": dataset_service.records(
            label=label,
            query=query,
        )
    }


@app.get("/api/records/{record_id}/beats")
def record_beats(
    record_id: str,
    source: str = "atr",
    label: str | None = None,
):
    try:
        return {
            "record_id": record_id,
            "source": source,
            "beats": dataset_service.record_beats(
                record_id,
                source=source,
                label=label,
            ),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


# Backward-compatible alias
@app.get("/api/records/{record_id}/segments")
def record_segments(
    record_id: str,
    label: str | None = None,
):
    return {
        "record_id": record_id,
        "segments": dataset_service.record_beats(
            record_id,
            source="test",
            label=label,
        ),
    }


@app.get("/api/records/{record_id}/waveform")
def record_waveform(
    record_id: str,
    channel: str | None = None,
    max_points: int = Query(default=5000, ge=500, le=20000),
):
    try:
        return dataset_service.record_waveform(
            record_id,
            channel,
            max_points,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc



@app.get("/api/records/{record_id}/window")
def record_window(
    record_id: str,
    channel: str | None = None,
    start_sec: float = Query(default=0.0),
    duration_sec: float = Query(default=10.0),
):
    try:
        return dataset_service.record_window(
            record_id=record_id,
            channel=channel,
            start_sec=start_sec,
            duration_sec=duration_sec,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.get("/api/beats/{beat_token:path}")
def beat_sample(beat_token: str):
    try:
        return dataset_service.load_beat(beat_token)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.get("/api/nas/current")
def nas_current():
    return model_service.summary()


@app.get("/api/candidate-jobs")
def candidate_jobs():
    return {
        "jobs":
            candidate_service.list_jobs()
    }


@app.get("/api/candidates")
def candidates(
    job_id: str | None = None,
):
    context = (
        candidate_service
        .get_context(
            job_id
        )
    )

    return {
        "count":
            len(
                context["candidates"]
            ),

        **context,
    }


@app.post("/api/candidates/select")
def select_candidate(
    req: CandidateSelectionRequest,
):
    try:
        return (
            candidate_service
            .select_candidate(
                candidate_id=
                    req.candidate_id,

                job_id=
                    req.job_id,
            )
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.get("/api/fpga/resources")
def fpga_resources():
    return fpga_resource_service.get()


@app.post("/api/infer/local")
def infer_local(req: BeatRequest):
    try:
        if req.beat_token:
            sample = dataset_service.load_beat(
                req.beat_token
            )
        elif req.sample_index is not None:
            sample = dataset_service.load_sample(
                req.sample_index
            )
        else:
            raise ValueError(
                "Request phải có beat_token hoặc sample_index."
            )

        start_time = time.perf_counter()

        result = inference_service.run(
            sample["normalized"]
        )

        total_local_ms = (time.perf_counter() - start_time) * 1000.0

        result["source"] = "LAPTOP_INTEGER"
        result["execution_mode"] = "LOCAL_INTEGER"
        result["timing"] = {"total_local_ms": float(total_local_ms)}
        result["true_label"] = sample["label"]
        result["record_id"] = sample["record_id"]
        result["beat_token"] = sample["token"]
        result["correct"] = (
            result["predicted_label"] == sample["label"]
        )

        return result
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post("/api/send")
def send_to_pynq(req: SendRequest):
    try:
        sample = dataset_service.load_beat(req.beat_token)

        return PynqService.send_sample(
            host=PYNQ_HOST,
            port=PYNQ_PORT,
            sample=sample,
            execution_mode=PYNQ_MODE,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post("/api/test/run")
def start_full_test(req: FullTestRequest):
    try:
        return batch_test_service.start(
            req.mode
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.get("/api/test/latest")
def latest_full_test():
    return {
        "job":
            batch_test_service.latest()
    }


@app.get("/api/test/jobs/{job_id}")
def full_test_job(job_id: str):
    job = batch_test_service.get(
        job_id
    )

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Không tìm thấy Full Test job.",
        )

    return job


@app.post("/api/test/jobs/{job_id}/cancel")
def cancel_full_test(job_id: str):
    try:
        return batch_test_service.cancel(
            job_id
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Không tìm thấy Full Test job.",
        ) from exc


@app.get("/api/nas/preflight")
def nas_preflight():
    split_dir = (
        dataset_service.test_csv.parent

        if dataset_service.test_csv

        else (
            PROJECT_ROOT
            / "datasets"
            / "splits"
        )
    )

    required = {
        "data_loader":
            PROJECT_ROOT
            / "data"
            / "loader.py",

        "nas_model":
            PROJECT_ROOT
            / "models"
            / "nas_model.py",

        "cost":
            PROJECT_ROOT
            / "nas"
            / "cost.py",

        "training":
            PROJECT_ROOT
            / "training"
            / "train.py",

        "train_csv":
            split_dir
            / "train.csv",

        "val_csv":
            split_dir
            / "val.csv",
    }

    files = {
        key: {
            "path":
                str(path),

            "exists":
                path.exists(),
        }

        for key, path
        in required.items()
    }

    packages = {
        "tensorflow":
            (
                importlib.util.find_spec(
                    "tensorflow"
                )
                is not None
            ),

        "sklearn":
            (
                importlib.util.find_spec(
                    "sklearn"
                )
                is not None
            ),

        "numpy":
            (
                importlib.util.find_spec(
                    "numpy"
                )
                is not None
            ),
    }

    return {
        "ok":
            (
                all(
                    item["exists"]
                    for item
                    in files.values()
                )
                and all(
                    packages.values()
                )
            ),

        "files":
            files,

        "packages":
            packages,

        "runtime": {
            "python":
                sys.executable,

            "pydantic":
                pydantic.__version__,
        },
    }


@app.post("/api/nas/search")
def start_nas_search(req: NasSearchRequest):
    if len(req.kernels) < 3:
        raise HTTPException(
            status_code=400,
            detail="Hãy chọn ít nhất 3 kernel cho 3 branch Inception.",
        )

    if not req.channels:
        raise HTTPException(
            status_code=400,
            detail="Phải nhập channel search space.",
        )


    config = (
        req.model_dump()
        if hasattr(req, "model_dump")
        else req.dict()
    )

    split_dir = (
        dataset_service.test_csv.parent
        if dataset_service.test_csv
        else PROJECT_ROOT / "datasets" / "splits"
    )

    config["fixed_target"] = {
        "device":
            "PYNQ-Z2 / Zynq-7020",

        "fpga_clock_mhz":
            40,

        "physical_resources":
            (
                "Fixed by current accelerator bitstream; "
                "LUT/BRAM/DSP are not candidate variables."
            ),

        "mixed_precision":
            (
                "Separate deployment step after architecture search."
            ),
    }

    config["dataset"] = {
        "train_csv": str(split_dir / "train.csv"),
        "val_csv": str(split_dir / "val.csv"),
        "test_csv": (
            str(dataset_service.test_csv)
            if dataset_service.test_csv
            else None
        ),
        "note": (
            "NAS dùng train.csv để train, val.csv để chọn model; "
            "test.csv chỉ dùng đánh giá cuối."
        ),
    }

    try:
        return nas_service.start(
            config
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"NAS start failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc


@app.get("/api/nas/jobs/{job_id}")
def nas_job(job_id: str):
    job = nas_service.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Không tìm thấy NAS job.",
        )

    return job
