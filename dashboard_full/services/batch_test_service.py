
from __future__ import annotations

import csv
import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np


LABELS = ["N", "L", "R", "V", "A"]


class BatchTestService:
    def __init__(
        self,
        project_root,
        dataset_service,
        inference_service,
        pynq_service,
        pynq_host,
        pynq_port,
        pynq_mode,
    ):
        self.root = Path(project_root)
        self.dataset_service = dataset_service
        self.inference_service = inference_service
        self.pynq_service = pynq_service
        self.pynq_host = pynq_host
        self.pynq_port = int(pynq_port)
        self.pynq_mode = pynq_mode

        self.jobs = {}
        self.lock = threading.Lock()

        self.output_dir = (
            self.root
            / "artifacts"
            / "test_runs"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _now_iso():
        return datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

    @staticmethod
    def _safe_float(value):
        if value is None:
            return None

        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _empty_matrix():
        return [
            [0 for _ in LABELS]
            for _ in LABELS
        ]

    @classmethod
    def _metrics_from_confusion(
        cls,
        confusion,
    ):
        matrix = np.asarray(
            confusion,
            dtype=np.int64,
        )

        support_total = int(
            matrix.sum()
        )

        correct = int(
            np.trace(matrix)
        )

        accuracy = (
            correct / support_total
            if support_total
            else 0.0
        )

        per_class = {}
        f1_values = []

        for i, label in enumerate(
            LABELS
        ):
            tp = int(matrix[i, i])
            fp = int(
                matrix[:, i].sum()
                - tp
            )
            fn = int(
                matrix[i, :].sum()
                - tp
            )
            support = int(
                matrix[i, :].sum()
            )

            precision = (
                tp / (tp + fp)
                if (tp + fp)
                else 0.0
            )

            recall = (
                tp / (tp + fn)
                if (tp + fn)
                else 0.0
            )

            f1 = (
                2.0
                * precision
                * recall
                / (precision + recall)
                if (precision + recall)
                else 0.0
            )

            f1_values.append(
                f1
            )

            per_class[label] = {
                "precision":
                    float(precision),

                "recall":
                    float(recall),

                "f1":
                    float(f1),

                "support":
                    support,
            }

        macro_f1 = float(
            np.mean(
                f1_values
            )
        )

        return {
            "accuracy":
                float(accuracy),

            "macro_f1":
                macro_f1,

            "correct":
                correct,

            "evaluated":
                support_total,

            "per_class":
                per_class,
        }

    def _snapshot(self, job):
        result = {
            key: value
            for key, value
            in job.items()
            if key not in {
                "_cancel_event",
                "_thread",
                "_latencies",
                "_fpga_latencies",
            }
        }

        confusion = result.get(
            "confusion_matrix",
            self._empty_matrix(),
        )

        result["metrics"] = (
            self._metrics_from_confusion(
                confusion
            )
        )

        latencies = job.get(
            "_latencies",
            [],
        )

        fpga_latencies = job.get(
            "_fpga_latencies",
            [],
        )

        result[
            "avg_total_latency_ms"
        ] = (
            float(
                np.mean(
                    latencies
                )
            )
            if latencies
            else None
        )

        result[
            "avg_fpga_weighted_ms"
        ] = (
            float(
                np.mean(
                    fpga_latencies
                )
            )
            if fpga_latencies
            else None
        )

        if (
            result[
                "avg_total_latency_ms"
            ]
            is not None
            and result[
                "avg_fpga_weighted_ms"
            ]
            is not None
        ):
            result[
                "avg_non_fpga_ms"
            ] = max(
                0.0,
                result[
                    "avg_total_latency_ms"
                ]
                - result[
                    "avg_fpga_weighted_ms"
                ],
            )

        else:
            result[
                "avg_non_fpga_ms"
            ] = None

        processed = int(
            result.get(
                "processed",
                0,
            )
        )

        total = int(
            result.get(
                "total",
                0,
            )
        )

        elapsed = float(
            result.get(
                "elapsed_seconds",
                0.0,
            )
            or 0.0
        )

        if (
            processed > 0
            and elapsed > 0.0
            and processed < total
        ):
            rate = processed / elapsed

            result[
                "eta_seconds"
            ] = (
                (total - processed)
                / rate
                if rate > 0
                else None
            )

        else:
            result[
                "eta_seconds"
            ] = 0.0 if processed >= total else None

        result["progress_percent"] = (
            100.0
            * processed
            / total
            if total
            else 0.0
        )

        return result

    def _active_job(self):
        for job in self.jobs.values():
            if job.get("status") in {
                "preparing",
                "running",
                "cancelling",
            }:
                return job

        return None

    def start(self, mode):
        mode = (
            "fpga"
            if str(mode).lower() == "fpga"
            else "local"
        )

        total = int(
            len(
                self.dataset_service.df
            )
        )

        if total <= 0:
            raise RuntimeError(
                "Không có sample trong test.csv."
            )

        with self.lock:
            active = self._active_job()

            if active is not None:
                raise RuntimeError(
                    "Đang có một Full Test job chạy. "
                    "Hãy chờ hoàn tất hoặc nhấn Stop."
                )

            job_id = (
                "test_"
                + datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
                + "_"
                + uuid.uuid4().hex[:6]
            )

            stamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            csv_path = (
                self.output_dir
                / f"{stamp}_{mode}_full_test.csv"
            )

            summary_path = (
                self.output_dir
                / f"{stamp}_{mode}_summary.json"
            )

            job = {
                "job_id":
                    job_id,

                "mode":
                    mode,

                "status":
                    "preparing",

                "total":
                    total,

                "processed":
                    0,

                "failed":
                    0,

                "started_at":
                    self._now_iso(),

                "finished_at":
                    None,

                "elapsed_seconds":
                    0.0,

                "current_index":
                    None,

                "current_token":
                    None,

                "current_true_label":
                    None,

                "current_predicted_label":
                    None,

                "last_error":
                    None,

                "confusion_matrix":
                    self._empty_matrix(),

                "output_csv":
                    str(
                        csv_path
                    ),

                "summary_json":
                    str(
                        summary_path
                    ),

                "_latencies":
                    [],

                "_fpga_latencies":
                    [],

                "_cancel_event":
                    threading.Event(),
            }

            thread = threading.Thread(
                target=self._run_job,
                args=(
                    job_id,
                    csv_path,
                    summary_path,
                ),
                daemon=True,
                name=job_id,
            )

            job["_thread"] = thread
            self.jobs[job_id] = job
            thread.start()

            return self._snapshot(
                job
            )

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(
                job_id
            )

            if job is None:
                raise KeyError(
                    job_id
                )

            if job.get("status") not in {
                "preparing",
                "running",
            }:
                return self._snapshot(
                    job
                )

            job["status"] = "cancelling"
            job["_cancel_event"].set()

            return self._snapshot(
                job
            )

    def get(self, job_id):
        with self.lock:
            job = self.jobs.get(
                job_id
            )

            if job is None:
                return None

            return self._snapshot(
                job
            )

    def latest(self):
        with self.lock:
            if not self.jobs:
                return None

            job = list(
                self.jobs.values()
            )[-1]

            return self._snapshot(
                job
            )

    def _update_elapsed(
        self,
        job,
        started,
    ):
        job["elapsed_seconds"] = (
            time.perf_counter()
            - started
        )

    def _run_local(
        self,
        sample,
    ):
        start = time.perf_counter()

        result = self.inference_service.run(
            sample["normalized"]
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        return {
            "predicted_label":
                result[
                    "predicted_label"
                ],

            "total_latency_ms":
                float(
                    elapsed_ms
                ),

            "fpga_weighted_ms":
                None,
        }

    def _run_fpga(
        self,
        sample,
    ):
        result = (
            self.pynq_service
            .send_sample(
                host=self.pynq_host,
                port=self.pynq_port,
                sample=sample,
                execution_mode=self.pynq_mode,
                timeout=180.0,
            )
        )

        if not result.get(
            "success"
        ):
            raise RuntimeError(
                result.get(
                    "detail"
                )
                or result.get(
                    "message"
                )
                or result.get(
                    "error"
                )
                or "PYNQ inference failed."
            )

        response = result.get(
            "response"
        ) or {}

        predicted = response.get(
            "predicted_label"
        )

        if predicted not in LABELS:
            raise RuntimeError(
                "PYNQ response không có predicted_label hợp lệ."
            )

        timing = response.get(
            "timing"
        ) or {}

        return {
            "predicted_label":
                predicted,

            "total_latency_ms":
                self._safe_float(
                    timing.get(
                        "total_hybrid_ms"
                    )
                ),

            "fpga_weighted_ms":
                self._safe_float(
                    timing.get(
                        "fpga_weighted_ms"
                    )
                ),
        }

    def _run_job(
        self,
        job_id,
        csv_path,
        summary_path,
    ):
        with self.lock:
            job = self.jobs[
                job_id
            ]

            job["status"] = "running"

        started = time.perf_counter()

        fieldnames = [
            "index",
            "beat_token",
            "record_id",
            "true_label",
            "predicted_label",
            "correct",
            "total_latency_ms",
            "fpga_weighted_ms",
            "error",
        ]

        try:
            with csv_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fieldnames,
                )

                writer.writeheader()

                total = int(
                    len(
                        self.dataset_service.df
                    )
                )

                for index in range(
                    total
                ):
                    with self.lock:
                        job = self.jobs[
                            job_id
                        ]

                        if job[
                            "_cancel_event"
                        ].is_set():
                            job[
                                "status"
                            ] = "cancelled"

                            break

                        job[
                            "current_index"
                        ] = index

                    sample = None

                    try:
                        sample = (
                            self.dataset_service
                            .load_sample(
                                index
                            )
                        )

                        true_label = str(
                            sample[
                                "label"
                            ]
                        )

                        with self.lock:
                            job = self.jobs[
                                job_id
                            ]

                            job[
                                "current_token"
                            ] = sample.get(
                                "token"
                            )

                            job[
                                "current_true_label"
                            ] = true_label

                        if job[
                            "mode"
                        ] == "fpga":
                            output = (
                                self._run_fpga(
                                    sample
                                )
                            )

                        else:
                            output = (
                                self._run_local(
                                    sample
                                )
                            )

                        predicted = str(
                            output[
                                "predicted_label"
                            ]
                        )

                        if (
                            true_label
                            not in LABELS
                            or predicted
                            not in LABELS
                        ):
                            raise ValueError(
                                "Label ngoài N/L/R/V/A."
                            )

                        true_id = LABELS.index(
                            true_label
                        )

                        pred_id = LABELS.index(
                            predicted
                        )

                        total_latency = (
                            output.get(
                                "total_latency_ms"
                            )
                        )

                        fpga_latency = (
                            output.get(
                                "fpga_weighted_ms"
                            )
                        )

                        with self.lock:
                            job = self.jobs[
                                job_id
                            ]

                            job[
                                "confusion_matrix"
                            ][true_id][pred_id] += 1

                            job[
                                "current_predicted_label"
                            ] = predicted

                            if total_latency is not None:
                                job[
                                    "_latencies"
                                ].append(
                                    float(
                                        total_latency
                                    )
                                )

                            if fpga_latency is not None:
                                job[
                                    "_fpga_latencies"
                                ].append(
                                    float(
                                        fpga_latency
                                    )
                                )

                        writer.writerow(
                            {
                                "index":
                                    index,

                                "beat_token":
                                    sample.get(
                                        "token"
                                    ),

                                "record_id":
                                    sample.get(
                                        "record_id"
                                    ),

                                "true_label":
                                    true_label,

                                "predicted_label":
                                    predicted,

                                "correct":
                                    int(
                                        true_label
                                        == predicted
                                    ),

                                "total_latency_ms":
                                    total_latency,

                                "fpga_weighted_ms":
                                    fpga_latency,

                                "error":
                                    "",
                            }
                        )

                    except Exception as exc:
                        with self.lock:
                            job = self.jobs[
                                job_id
                            ]

                            job[
                                "failed"
                            ] += 1

                            job[
                                "last_error"
                            ] = (
                                f"{type(exc).__name__}: {exc}"
                            )

                        writer.writerow(
                            {
                                "index":
                                    index,

                                "beat_token":
                                    (
                                        sample.get(
                                            "token"
                                        )
                                        if sample
                                        else ""
                                    ),

                                "record_id":
                                    (
                                        sample.get(
                                            "record_id"
                                        )
                                        if sample
                                        else ""
                                    ),

                                "true_label":
                                    (
                                        sample.get(
                                            "label"
                                        )
                                        if sample
                                        else ""
                                    ),

                                "predicted_label":
                                    "",

                                "correct":
                                    "",

                                "total_latency_ms":
                                    "",

                                "fpga_weighted_ms":
                                    "",

                                "error":
                                    f"{type(exc).__name__}: {exc}",
                            }
                        )

                    finally:
                        with self.lock:
                            job = self.jobs[
                                job_id
                            ]

                            job[
                                "processed"
                            ] = index + 1

                            self._update_elapsed(
                                job,
                                started,
                            )

                        handle.flush()

            with self.lock:
                job = self.jobs[
                    job_id
                ]

                if job[
                    "status"
                ] not in {
                    "cancelled",
                    "failed",
                }:
                    job[
                        "status"
                    ] = "completed"

                job[
                    "finished_at"
                ] = self._now_iso()

                self._update_elapsed(
                    job,
                    started,
                )

                snapshot = self._snapshot(
                    job
                )

            summary_path.write_text(
                json.dumps(
                    snapshot,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        except Exception as exc:
            with self.lock:
                job = self.jobs[
                    job_id
                ]

                job[
                    "status"
                ] = "failed"

                job[
                    "finished_at"
                ] = self._now_iso()

                job[
                    "last_error"
                ] = (
                    f"{type(exc).__name__}: {exc}"
                )

                self._update_elapsed(
                    job,
                    started,
                )

                snapshot = self._snapshot(
                    job
                )

            try:
                summary_path.write_text(
                    json.dumps(
                        snapshot,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass
