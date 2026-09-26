from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"

# NAS configs are stored only at the project root:
#   ECG_NAS_main/configs/*.json
CONFIGS_DIR = PROJECT_ROOT / "configs"
ORIGINAL_CONFIG_NAME = "evolutionary_nas_final"
PREFERRED_CONFIG_NAMES = (
    "evolutionary_nas_searchspace_v2",
    ORIGINAL_CONFIG_NAME,
)

NAS_RUN_ROOT = PROJECT_ROOT / "artifacts" / "nas_runs"
BENCH_DIR = PROJECT_ROOT / "artifacts" / "dashboard_benchmarks"

CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
NAS_RUN_ROOT.mkdir(parents=True, exist_ok=True)
BENCH_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

LABELS = ["N", "L", "R", "V", "A"]
LABEL_TO_ID = {v: i for i, v in enumerate(LABELS)}
INTERPRETER_CACHE: dict[int, Any] = {}

NAS_LOCK = threading.Lock()
NAS_PROCESS: subprocess.Popen | None = None
NAS_JOB: dict[str, Any] = {
    "state": "IDLE",
    "progress": 0.0,
    "message": "No NAS job has been started from this dashboard.",
    "config_name": "",
    "active_seed": None,
    "completed_candidates": 0,
    "total_candidates": 0,
    "output_dir": "",
    "started_at": None,
    "finished_at": None,
    "pid": None,
    "error": None,
    "cancel_requested": False,
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def candidates():
    return load_json(DATA_DIR / "candidates.json")


def candidate_by_id(cid: int):
    for c in candidates():
        if int(c["candidate_id"]) == int(cid):
            return c
    raise KeyError(cid)


def normalize_beat(x, length=320):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) > length:
        center = len(x) // 2
        start = max(0, center - length // 2)
        x = x[start:start + length]
    elif len(x) < length:
        left = (length - len(x)) // 2
        right = length - len(x) - left
        x = np.pad(x, (left, right), mode="constant")
    std = float(x.std())
    if std < 1e-6:
        std = 1.0
    return ((x - float(x.mean())) / std).astype(np.float32)


def downsample(x, n=56):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) <= n:
        return [float(v) for v in x]
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return [round(float(v), 5) for v in x[idx]]


def _valid_csv_dir(path: Path | None):
    return bool(
        path
        and path.is_dir()
        and (path / "train.csv").exists()
        and (path / "val.csv").exists()
        and (path / "test.csv").exists()
    )


def _valid_record_dir(path: Path | None):
    if not path or not path.is_dir():
        return False
    try:
        return any(path.glob("*.dat")) and any(path.glob("*.hea"))
    except Exception:
        return False


def configured_dataset_paths():
    """Find the project dataset automatically. No dashboard Settings screen is required."""
    preferred = PROJECT_ROOT / "datasets"
    if _valid_csv_dir(preferred) and _valid_record_dir(preferred):
        return preferred.resolve(), preferred.resolve()

    roots = [PROJECT_ROOT, PROJECT_ROOT.parent, BASE_DIR, Path.cwd()]
    csv_hits: list[Path] = []
    record_hits: list[Path] = []
    seen: set[Path] = set()

    for root in roots:
        try:
            root = root.resolve()
        except Exception:
            continue
        if root in seen or not root.exists():
            continue
        seen.add(root)

        for sub in [root, root / "datasets", root / "dataset", root / "data"]:
            if not sub.exists() or not sub.is_dir():
                continue
            try:
                if _valid_csv_dir(sub):
                    csv_hits.append(sub)
                else:
                    for p in sub.rglob("train.csv"):
                        if _valid_csv_dir(p.parent):
                            csv_hits.append(p.parent)
                            break

                if _valid_record_dir(sub):
                    record_hits.append(sub)
                else:
                    for cur, _dirs, _files in os.walk(sub):
                        cur_path = Path(cur)
                        if _valid_record_dir(cur_path):
                            record_hits.append(cur_path)
                            break
            except Exception:
                pass

    return (
        csv_hits[0].resolve() if csv_hits else None,
        record_hits[0].resolve() if record_hits else None,
    )


def dataset_status():
    csv_dir, record_dir = configured_dataset_paths()
    connected = bool(csv_dir and record_dir)
    return {
        "connected": connected,
        "csv_dir": str(csv_dir) if csv_dir else "",
        "record_dir": str(record_dir) if record_dir else "",
        "mode": "MIT-BIH" if connected else "NOT CONNECTED",
    }


def actual_records(split: str):
    csv_dir, _ = configured_dataset_paths()
    if not csv_dir:
        raise RuntimeError("MIT-BIH dataset was not found under the project datasets folder")

    csv_path = csv_dir / f"{split}.csv"
    if not csv_path.exists():
        raise RuntimeError(f"Missing dataset split: {csv_path}")

    df = pd.read_csv(csv_path)
    if "record_path" not in df.columns or "labels" not in df.columns:
        raise RuntimeError("Dataset CSV must contain record_path and labels columns")

    rows = []
    ids = df["record_path"].astype(str).map(lambda s: Path(s.replace("\\", "/")).stem)
    df = df.assign(_record_id=ids)
    for rid, g in df.groupby("_record_id"):
        counts = {label: int((g["labels"].astype(str) == label).sum()) for label in LABELS}
        rows.append({
            "record_id": str(rid),
            "split": split,
            "beat_count": int(len(g)),
            "class_counts": counts,
        })
    return sorted(rows, key=lambda x: x["record_id"])


def actual_record(split: str, record_id: str):
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("wfdb is required. Install it with: pip install wfdb") from exc

    csv_dir, record_dir = configured_dataset_paths()
    if not csv_dir or not record_dir:
        raise RuntimeError("MIT-BIH dataset was not found")

    csv_path = csv_dir / f"{split}.csv"
    df = pd.read_csv(csv_path)
    ids = df["record_path"].astype(str).map(lambda s: Path(s.replace("\\", "/")).stem)
    df = df.loc[ids == record_id].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"Record {record_id} not found in {split}.csv")

    base = record_dir / record_id
    signal, fields = wfdb.rdsamp(str(base))
    sig_names = fields.get("sig_name", [])

    beats = []
    for i, row in df.iterrows():
        ch_name = str(row.get("channel", "MLII"))
        ch = sig_names.index(ch_name) if ch_name in sig_names else 0
        start = int(row["start"])
        end = int(row["end"])
        full = normalize_beat(signal[start:end + 1, ch])
        label = str(row["labels"])
        beats.append({
            "index": int(i),
            "label": label,
            "start": start,
            "end": end,
            "points": downsample(full, 56),
            "samples": [round(float(v), 6) for v in full],
        })

    class_counts = {label: int((df["labels"].astype(str) == label).sum()) for label in LABELS}
    return {
        "record_id": record_id,
        "split": split,
        "beat_count": int(len(df)),
        "beats_loaded": len(beats),
        "truncated": False,
        "class_counts": class_counts,
        "beats": beats,
        "source": "MIT-BIH",
    }


def find_quantized_model(cid: int):
    token = f"{int(cid):04d}"
    roots = [
        PROJECT_ROOT / "artifacts" / "quantized_int8",
        PROJECT_ROOT / "quantized_int8",
        PROJECT_ROOT / "artifacts",
        PROJECT_ROOT,
    ]
    seen: set[Path] = set()
    for root in roots:
        try:
            root = root.resolve()
        except Exception:
            continue
        if root in seen or not root.exists():
            continue
        seen.add(root)
        try:
            for model_path in root.rglob("model_int8.tflite"):
                text = str(model_path).lower()
                if token in text or f"candidate_{int(cid)}" in text:
                    return model_path.resolve()
        except Exception:
            pass
    return None


def tensorflow_available():
    try:
        import tensorflow as _tf  # noqa: F401
        return True
    except Exception:
        return False


def model_runtime_status():
    paths = {}
    for c in candidates():
        cid = int(c["candidate_id"])
        model_path = find_quantized_model(cid)
        paths[str(cid)] = str(model_path) if model_path else ""
    return {
        "tensorflow_available": tensorflow_available(),
        "models_found": sum(bool(v) for v in paths.values()),
        "models_total": len(paths),
        "model_paths": paths,
    }


def tflite_predict(cid: int, samples: list[float]):
    model_path = find_quantized_model(cid)
    if not model_path:
        raise RuntimeError(f"INT8 model for Candidate {int(cid):04d} was not found under artifacts/quantized_int8")
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError("TensorFlow is not installed in the active virtual environment") from exc

    if cid not in INTERPRETER_CACHE:
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        INTERPRETER_CACHE[cid] = interpreter

    interpreter = INTERPRETER_CACHE[cid]
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    x = normalize_beat(samples).reshape(1, 320, 1).astype(np.float32)
    in_scale, in_zero = inp["quantization"]
    if inp["dtype"] == np.int8:
        if not in_scale:
            raise RuntimeError("Invalid TFLite input quantization scale")
        xq = np.round(x / in_scale + in_zero)
        xq = np.clip(xq, -128, 127).astype(np.int8)
    else:
        xq = x.astype(inp["dtype"])

    t0 = time.perf_counter_ns()
    interpreter.set_tensor(inp["index"], xq)
    interpreter.invoke()
    yq = interpreter.get_tensor(out["index"])
    t1 = time.perf_counter_ns()

    out_scale, out_zero = out["quantization"]
    if out["dtype"] == np.int8:
        probs = (yq.astype(np.float32) - out_zero) * out_scale
    else:
        probs = yq.astype(np.float32)

    probs = np.maximum(probs.reshape(-1), 0)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    pred = int(np.argmax(probs))

    return {
        "backend": "TFLite INT8",
        "predicted_class": LABELS[pred],
        "confidence": float(probs[pred]),
        "probabilities": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))},
        "latency_ms": (t1 - t0) / 1e6,
        "model_path": str(model_path),
    }


def cpu_predict(cid: int, beat: dict[str, Any]):
    return tflite_predict(cid, beat.get("samples", []))


def pynq_endpoint():
    return str(os.environ.get("PYNQ_ENDPOINT", "")).strip()


def fpga_predict(cid: int, beat: dict[str, Any]):
    endpoint = pynq_endpoint()
    if not endpoint:
        raise RuntimeError(
            "PYNQ endpoint is not configured. Set environment variable PYNQ_ENDPOINT, "
            "for example: $env:PYNQ_ENDPOINT='http://192.168.x.x:8000/infer'"
        )

    payload = {
        "candidate_id": cid,
        "samples": beat.get("samples", []),
        "reference_label": beat.get("label"),
    }
    t0 = time.perf_counter_ns()
    response = requests.post(endpoint, json=payload, timeout=60)
    response.raise_for_status()
    out = response.json()
    t1 = time.perf_counter_ns()
    out.setdefault("latency_ms", (t1 - t0) / 1e6)
    out.setdefault("backend", "PYNQ-Z2")
    return out


def default_search_space():
    return {
        "kernel_choices": [[3, 5, 7], [3, 5, 9], [3, 7, 9], [5, 7, 9]],
        "channel_choices": list(range(4, 65, 4)),
    }


def normalize_config_for_dashboard(payload):
    """Normalize old/new config files for the dashboard UI.

    Existing project configs may contain a single ``seed`` value and may not
    yet contain ``search_space``. The dashboard keeps those files readable
    without modifying them on disk.
    """
    cfg = dict(payload)

    if "seeds" not in cfg:
        if "seed" in cfg:
            cfg["seeds"] = [int(cfg["seed"])]
        else:
            cfg["seeds"] = [42]
    elif isinstance(cfg["seeds"], (int, float, str)):
        cfg["seeds"] = [cfg["seeds"]]

    # The dashboard manages one or more seeds; the per-seed worker writes
    # ``seed`` into each generated run config.
    cfg.pop("seed", None)

    if not cfg.get("search_space"):
        cfg["search_space"] = default_search_space()

    return cfg


def validate_config(payload):
    payload = normalize_config_for_dashboard(payload)

    int_fields = [
        "population_size", "generations", "elite_count", "tournament_size",
        "epochs_per_candidate", "batch_size", "early_stop_patience",
        "max_params", "max_macs", "max_proposal_attempts",
    ]
    float_fields = [
        "mutation_rate", "crossover_rate", "learning_rate", "max_memory_mb", "f1_tolerance",
    ]
    cfg: dict[str, Any] = {}
    for key in int_fields:
        cfg[key] = int(payload[key])
    for key in float_fields:
        cfg[key] = float(payload[key])

    seeds = payload.get("seeds", [42])
    if isinstance(seeds, str):
        seeds = [int(x.strip()) for x in seeds.split(",") if x.strip()]
    cfg["seeds"] = [int(x) for x in seeds]

    raw_ss = payload.get("search_space") or default_search_space()
    kernels = []
    seen_kernels = set()
    for item in raw_ss.get("kernel_choices", default_search_space()["kernel_choices"]):
        values = tuple(int(x) for x in item)
        if len(values) != 3 or any(x <= 0 for x in values):
            raise ValueError("Each kernel choice must contain exactly 3 positive integers")
        if values not in seen_kernels:
            kernels.append(list(values))
            seen_kernels.add(values)

    channels = sorted({
        int(x)
        for x in raw_ss.get("channel_choices", default_search_space()["channel_choices"])
    })
    if not kernels:
        raise ValueError("Select at least one kernel triplet")
    if not channels or any(x <= 0 for x in channels):
        raise ValueError("Select at least one positive channel width")

    cfg["search_space"] = {
        "kernel_choices": kernels,
        "channel_choices": channels,
    }

    if cfg["population_size"] < 2 or cfg["generations"] < 1:
        raise ValueError("Population must be >= 2 and generations >= 1")
    if cfg["elite_count"] < 1 or cfg["elite_count"] >= cfg["population_size"]:
        raise ValueError("Elite count must be >= 1 and smaller than population size")
    if not 0 <= cfg["mutation_rate"] <= 1 or not 0 <= cfg["crossover_rate"] <= 1:
        raise ValueError("Mutation/crossover rates must be within [0,1]")
    if not cfg["seeds"]:
        raise ValueError("At least one random seed is required")
    return cfg


def _config_file_for_name(name: str):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not safe:
        raise ValueError("Invalid config name")
    return CONFIGS_DIR / f"{safe}.json"


def _config_paths():
    return sorted(
        CONFIGS_DIR.glob("*.json"),
        key=lambda p: (p.name.lower(), p.stat().st_mtime),
    )


def preferred_config_name():
    names = {p.stem for p in _config_paths()}
    for name in PREFERRED_CONFIG_NAMES:
        if name in names:
            return name
    paths = _config_paths()
    return paths[0].stem if paths else ""


def list_saved_configs():
    items = []
    preferred = preferred_config_name()

    for path in _config_paths():
        try:
            raw = load_json(path)
            items.append({
                "name": path.stem,
                "filename": path.name,
                "path": str(path),
                "is_original": path.stem == ORIGINAL_CONFIG_NAME,
                "is_preferred": path.stem == preferred,
                "config": normalize_config_for_dashboard(raw),
            })
        except Exception:
            continue

    return items


def next_config_name():
    used = {item["name"] for item in list_saved_configs()}
    i = 1
    while f"config{i}" in used:
        i += 1
    return f"config{i}"


def unique_config_name(requested: str | None):
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", (requested or "").strip()).strip("._")
    if not base or base == ORIGINAL_CONFIG_NAME:
        base = next_config_name()

    used = {item["name"] for item in list_saved_configs()}
    if base not in used:
        return base

    i = 2
    while f"{base}_{i}" in used:
        i += 1
    return f"{base}_{i}"


def configurable_search_space_ready():
    """Check that the project NAS backend actually consumes search_space."""
    pruned = PROJECT_ROOT / "nas" / "pruned_search.py"
    runner = PROJECT_ROOT / "scripts" / "run_pruned_evolutionary_nas.py"

    if not pruned.exists() or not runner.exists():
        return False

    try:
        pruned_text = pruned.read_text(encoding="utf-8", errors="ignore")
        runner_text = runner.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False

    return (
        "normalize_search_space_config" in pruned_text
        and "random_architecture(rng, search_space)" in runner_text
        and "theoretical_search_space_size(search_space)" in runner_text
    )


def find_nas_script():
    candidates_to_check = [
        PROJECT_ROOT / "scripts" / "run_pruned_evolutionary_nas.py",
        PROJECT_ROOT / "run_pruned_evolutionary_nas.py",
        PROJECT_ROOT.parent / "scripts" / "run_pruned_evolutionary_nas.py",
        BASE_DIR / "run_pruned_evolutionary_nas.py",
    ]
    for path in candidates_to_check:
        if path.exists():
            return path.resolve()
    return None


def expected_candidates_per_seed(cfg: dict[str, Any]):
    population = int(cfg["population_size"])
    generations = int(cfg["generations"])
    elite = int(cfg["elite_count"])
    return population + max(0, generations - 1) * max(0, population - elite)


def _update_nas_job(**kwargs):
    with NAS_LOCK:
        NAS_JOB.update(kwargs)
        output = NAS_JOB.get("output_dir")
        if output:
            try:
                save_json(Path(output) / "dashboard_job_status.json", NAS_JOB)
            except Exception:
                pass


def _count_completed_candidates(seed_dir: Path):
    if not seed_dir.exists():
        return 0
    try:
        return len(list(seed_dir.glob("candidate_*/metrics.json")))
    except Exception:
        return 0


def _terminate_process(proc: subprocess.Popen | None):
    """Terminate the active NAS process, including its child tree on Windows."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
                check=False,
            )
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _cancel_requested():
    with NAS_LOCK:
        return bool(NAS_JOB.get("cancel_requested"))


def _nas_worker(config_name: str, cfg: dict[str, Any], run_dir: Path, script: Path):
    seeds = list(cfg["seeds"])
    per_seed = expected_candidates_per_seed(cfg)
    total = per_seed * len(seeds)
    started = datetime.now().isoformat(timespec="seconds")
    _update_nas_job(
        state="RUNNING", progress=0.0, message="NAS started", config_name=config_name,
        active_seed=seeds[0] if seeds else None, completed_candidates=0,
        total_candidates=total, output_dir=str(run_dir), started_at=started,
        finished_at=None, pid=None, error=None, cancel_requested=False,
    )

    global NAS_PROCESS

    try:
        for seed_index, seed in enumerate(seeds):
            seed_dir = run_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            seed_cfg = dict(cfg)
            seed_cfg.pop("seeds", None)
            seed_cfg["seed"] = int(seed)
            cfg_path = run_dir / f"config_seed_{seed}.json"
            save_json(cfg_path, seed_cfg)
            log_path = run_dir / f"seed_{seed}.log"

            cmd = [
                sys.executable,
                str(script),
                "--config", str(cfg_path),
                "--output", str(seed_dir),
            ]
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(script.parent),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                with NAS_LOCK:
                    NAS_PROCESS = proc
                _update_nas_job(
                    active_seed=int(seed), pid=proc.pid,
                    message=f"Running seed {seed} ({seed_index + 1}/{len(seeds)})",
                )

                while proc.poll() is None:
                    if _cancel_requested():
                        _terminate_process(proc)
                        with NAS_LOCK:
                            NAS_PROCESS = None
                        _update_nas_job(
                            state="CANCELLED", pid=None, active_seed=None,
                            message="NAS run cancelled by user. Partial results were kept in the run folder.",
                            finished_at=datetime.now().isoformat(timespec="seconds"),
                            cancel_requested=False, error=None,
                        )
                        return

                    current = min(_count_completed_candidates(seed_dir), per_seed)
                    completed = seed_index * per_seed + current
                    progress = 100.0 * completed / max(total, 1)
                    _update_nas_job(
                        completed_candidates=completed,
                        progress=round(progress, 2),
                        message=f"Seed {seed}: {current}/{per_seed} candidate evaluations completed",
                    )
                    time.sleep(1.0)

                return_code = proc.returncode
                with NAS_LOCK:
                    NAS_PROCESS = None

            if _cancel_requested():
                _update_nas_job(
                    state="CANCELLED", pid=None, active_seed=None,
                    message="NAS run cancelled by user. Partial results were kept in the run folder.",
                    finished_at=datetime.now().isoformat(timespec="seconds"),
                    cancel_requested=False, error=None,
                )
                return

            if return_code != 0:
                raise RuntimeError(
                    f"NAS process failed for seed {seed} with exit code {return_code}. "
                    f"Check {log_path}"
                )

            completed = (seed_index + 1) * per_seed
            progress = 100.0 * completed / max(total, 1)
            _update_nas_job(
                completed_candidates=completed,
                progress=round(progress, 2),
                message=f"Seed {seed} completed",
            )

        _update_nas_job(
            state="DONE", progress=100.0, completed_candidates=total,
            active_seed=None, pid=None,
            message="NAS completed successfully. Results were saved in a separate dashboard run folder.",
            finished_at=datetime.now().isoformat(timespec="seconds"), error=None, cancel_requested=False,
        )
    except Exception as exc:
        with NAS_LOCK:
            NAS_PROCESS = None
        if _cancel_requested():
            _update_nas_job(
                state="CANCELLED", pid=None, active_seed=None,
                message="NAS run cancelled by user. Partial results were kept in the run folder.",
                error=None, cancel_requested=False,
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )
        else:
            _update_nas_job(
                state="ERROR", pid=None, active_seed=None,
                message=str(exc), error=str(exc), cancel_requested=False,
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def api_status():
    return jsonify({
        "dataset": dataset_status(),
        "pynq_configured": bool(pynq_endpoint()),
        "runtime": model_runtime_status(),
        "nas": {
            "config_dir": str(CONFIGS_DIR),
            "search_space_backend_ready": configurable_search_space_ready(),
        },
    })


@app.get("/api/candidates")
def api_candidates():
    return jsonify(candidates())


@app.get("/api/dataset/records")
def api_records():
    split = request.args.get("split", "test")
    try:
        return jsonify({"source": "MIT-BIH", "records": actual_records(split)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503


@app.get("/api/dataset/record/<split>/<record_id>")
def api_record(split, record_id):
    try:
        return jsonify(actual_record(split, record_id))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/nas/configs")
def api_configs_list():
    configs = list_saved_configs()
    return jsonify({
        "configs": configs,
        "config_dir": str(CONFIGS_DIR),
        "preferred_name": preferred_config_name(),
    })


@app.get("/api/nas/config/<name>")
def api_config_get(name):
    try:
        path = _config_file_for_name(name)
        if not path.exists():
            return jsonify({"error": f"Config {name} not found in {CONFIGS_DIR}"}), 404
        return jsonify({
            "name": name,
            "is_original": name == ORIGINAL_CONFIG_NAME,
            "filename": path.name,
            "path": str(path),
            "config": normalize_config_for_dashboard(load_json(path)),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/nas/config/save")
def api_config_save_new():
    try:
        payload = request.get_json(force=True)
        cfg = validate_config(payload.get("config", payload))
        name = unique_config_name(payload.get("name"))
        path = _config_file_for_name(name)
        save_json(path, cfg)
        return jsonify({"ok": True, "name": name, "filename": path.name, "config": cfg})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/nas/run")
def api_nas_run():
    with NAS_LOCK:
        if NAS_JOB.get("state") == "RUNNING":
            return jsonify({"ok": False, "error": "A NAS job is already running"}), 409

    try:
        payload = request.get_json(force=True)
        cfg = validate_config(payload.get("config", payload))
        config_name = str(payload.get("config_name") or "unsaved")
        script = find_nas_script()
        if not script:
            return jsonify({
                "ok": False,
                "error": "run_pruned_evolutionary_nas.py was not found in the project/scripts folder",
            }), 400

        if not configurable_search_space_ready():
            return jsonify({
                "ok": False,
                "error": (
                    "The configurable search-space NAS patch is not active. "
                    "Replace nas/pruned_search.py and scripts/run_pruned_evolutionary_nas.py "
                    "with the patched versions before running NAS from this dashboard."
                ),
            }), 400

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", config_name).strip("_") or "config"
        run_dir = NAS_RUN_ROOT / f"run_{stamp}_{safe_name}"
        run_dir.mkdir(parents=True, exist_ok=False)
        save_json(run_dir / "selected_config_snapshot.json", cfg)
        save_json(run_dir / "run_metadata.json", {
            "config_name": config_name,
            "script": str(script),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "seeds": cfg["seeds"],
        })

        _update_nas_job(cancel_requested=False)
        thread = threading.Thread(
            target=_nas_worker,
            args=(config_name, cfg, run_dir, script),
            daemon=True,
        )
        thread.start()
        return jsonify({
            "ok": True,
            "state": "RUNNING",
            "output_dir": str(run_dir),
            "config_name": config_name,
            "script": str(script),
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/nas/cancel")
def api_nas_cancel():
    with NAS_LOCK:
        state = NAS_JOB.get("state")
        if state != "RUNNING":
            return jsonify({
                "ok": False,
                "error": f"No active NAS run to cancel (state={state})",
            }), 409
        NAS_JOB["cancel_requested"] = True
        NAS_JOB["message"] = "Cancellation requested. Stopping the active NAS process..."
        output = NAS_JOB.get("output_dir")
        if output:
            try:
                save_json(Path(output) / "dashboard_job_status.json", NAS_JOB)
            except Exception:
                pass
    return jsonify({"ok": True, "state": "CANCELLING"})


@app.get("/api/nas/status")
def api_nas_status():
    with NAS_LOCK:
        return jsonify(dict(NAS_JOB))


@app.post("/api/predict/software")
def api_predict_software():
    payload = request.get_json(force=True)
    cid = int(payload["candidate_id"])
    beat = payload["beat"]
    try:
        result = cpu_predict(cid, beat)
        result["candidate"] = candidate_by_id(cid)
        return jsonify({"ok": True, "result": result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/predict/fpga")
def api_predict_fpga():
    payload = request.get_json(force=True)
    cid = int(payload["candidate_id"])
    beat = payload["beat"]
    try:
        result = fpga_predict(cid, beat)
        result["candidate"] = candidate_by_id(cid)
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "candidate_id": cid,
            "record_id": payload.get("record_id"),
            "beat_index": beat.get("index"),
            "result": result,
        }
        log_path = BENCH_DIR / "fpga_results.json"
        rows = load_json(log_path) if log_path.exists() else []
        rows.append(row)
        save_json(log_path, rows[-500:])
        return jsonify({"ok": True, "result": result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/compare/<mode>")
def api_compare(mode):
    if mode not in {"software", "fpga"}:
        return jsonify({"error": "Unsupported comparison mode"}), 400

    payload = request.get_json(force=True)
    beat = payload["beat"]
    rows = []
    for c in candidates():
        cid = int(c["candidate_id"])
        try:
            result = fpga_predict(cid, beat) if mode == "fpga" else cpu_predict(cid, beat)
            rows.append({"candidate": c, "result": result, "ok": True})
        except Exception as exc:
            rows.append({"candidate": c, "ok": False, "error": str(exc)})
    return jsonify({"ok": True, "mode": mode, "rows": rows})


if __name__ == "__main__":
    print("ECG NAS Dashboard V7")
    print("Open: http://127.0.0.1:5000")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset: {dataset_status()}")
    print(f"INT8 runtime: {model_runtime_status()}")
    if not pynq_endpoint():
        print("PYNQ: not configured. Set environment variable PYNQ_ENDPOINT when FPGA inference service is ready.")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
