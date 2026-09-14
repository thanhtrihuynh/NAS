import csv
import json
from datetime import datetime
from pathlib import Path


class CandidateService:
    def __init__(self, project_root):
        self.root = Path(project_root)

        self.jobs_dir = (
            self.root
            / "artifacts"
            / "dashboard_nas_jobs"
        )

    @staticmethod
    def _candidate_id(item, fallback):
        for key in (
            "id",
            "candidate_id",
            "trial",
            "trial_id",
        ):
            if key in item:
                return item[key]

        return fallback

    @staticmethod
    def _pick(item, metrics, *keys):
        for key in keys:
            if key in item:
                return item[key]

            if (
                isinstance(metrics, dict)
                and key in metrics
            ):
                return metrics[key]

        return None

    def _normalize_list(
        self,
        data,
        source,
        job_id=None,
        selected_candidate_id=None,
    ):
        if isinstance(data, list):
            items = data

        elif isinstance(data, dict):
            items = None

            for key in (
                "candidates",
                "trials",
                "results",
                "history",
                "search_results",
            ):
                value = data.get(key)

                if isinstance(value, list):
                    items = value
                    break

            if items is None:
                return []

        else:
            return []

        normalized = []

        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            metrics = item.get(
                "metrics",
                {},
            )

            candidate_id = self._candidate_id(
                item,
                i,
            )

            architecture = (
                item.get("architecture")
                or item.get("arch")
                or item.get("config")
                or item.get("model_config")
                or {}
            )

            normalized.append(
                {
                    "id":
                        candidate_id,

                    "job_id":
                        job_id,

                    "status":
                        item.get("status")
                        or item.get("state")
                        or "unknown",

                    "macro_f1":
                        self._pick(
                            item,
                            metrics,
                            "macro_f1",
                            "f1_macro",
                            "f1",
                        ),

                    "accuracy":
                        self._pick(
                            item,
                            metrics,
                            "accuracy",
                            "acc",
                        ),

                    "params":
                        self._pick(
                            item,
                            metrics,
                            "params",
                            "parameters",
                            "n_params",
                        ),

                    "macs":
                        self._pick(
                            item,
                            metrics,
                            "macs",
                            "MACs",
                        ),

                    "memory_mb_fp32":
                        self._pick(
                            item,
                            metrics,
                            "memory_mb_fp32",
                        ),

                    "epochs_ran":
                        self._pick(
                            item,
                            metrics,
                            "epochs_ran",
                        ),

                    "best_val_loss":
                        self._pick(
                            item,
                            metrics,
                            "best_val_loss",
                        ),

                    "checkpoint_available":
                        bool(
                            item.get(
                                "checkpoint_available",
                                False,
                            )
                        ),

                    "checkpoint_path":
                        item.get(
                            "checkpoint_path"
                        ),

                    "candidate_dir":
                        item.get(
                            "candidate_dir"
                        ),

                    "config_path":
                        item.get(
                            "config_path"
                        ),

                    "history_path":
                        item.get(
                            "history_path"
                        ),

                    "metrics_path":
                        item.get(
                            "metrics_path"
                        ),

                    "score":
                        self._pick(
                            item,
                            metrics,
                            "score",
                            "fitness",
                        ),

                    "is_best":
                        bool(
                            item.get(
                                "is_best",
                                False,
                            )
                            or item.get(
                                "best",
                                False,
                            )
                        ),

                    "is_selected":
                        (
                            selected_candidate_id
                            is not None
                            and str(candidate_id)
                            == str(
                                selected_candidate_id
                            )
                        ),

                    "architecture":
                        architecture,

                    "error":
                        item.get("error"),

                    "source":
                        source,
                }
            )

        return normalized

    @staticmethod
    def _read_json(path):
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    def _selected_id(
        self,
        job_dir,
    ):
        path = (
            job_dir
            / "selected_candidate.json"
        )

        if not path.exists():
            return None

        try:
            data = self._read_json(
                path
            )

            return data.get(
                "candidate_id"
            )
        except Exception:
            return None

    def _job_candidates(
        self,
        job_dir,
    ):
        path = (
            job_dir
            / "candidates.json"
        )

        if not path.exists():
            return []

        try:
            data = self._read_json(
                path
            )

            return self._normalize_list(
                data,
                str(path),
                job_id=job_dir.name,
                selected_candidate_id=
                    self._selected_id(
                        job_dir
                    ),
            )
        except Exception:
            return []

    def list_jobs(self):
        if not self.jobs_dir.exists():
            return []

        jobs = []

        for job_dir in self.jobs_dir.iterdir():
            if not job_dir.is_dir():
                continue

            candidates = self._job_candidates(
                job_dir
            )

            if not candidates:
                continue

            status = "unknown"

            status_file = (
                job_dir
                / "status.json"
            )

            if status_file.exists():
                try:
                    status = (
                        self._read_json(
                            status_file
                        ).get(
                            "status",
                            "unknown",
                        )
                    )
                except Exception:
                    pass

            selected_id = self._selected_id(
                job_dir
            )

            best_candidate = next(
                (
                    item
                    for item in candidates
                    if item["is_best"]
                ),
                None,
            )

            jobs.append(
                {
                    "job_id":
                        job_dir.name,

                    "status":
                        status,

                    "candidate_count":
                        len(candidates),

                    "selected_candidate_id":
                        selected_id,

                    "best_candidate_id":
                        (
                            best_candidate["id"]
                            if best_candidate
                            else None
                        ),
                }
            )

        jobs.sort(
            key=lambda item:
                item["job_id"],
            reverse=True,
        )

        return jobs

    def latest_job_id(self):
        jobs = self.list_jobs()

        return (
            jobs[0]["job_id"]
            if jobs
            else None
        )

    def get_context(
        self,
        job_id=None,
    ):
        if job_id is None:
            job_id = self.latest_job_id()

        if job_id is None:
            return {
                "job_id": None,
                "selected_candidate_id": None,
                "best_candidate_id": None,
                "candidates": [],
            }

        job_dir = (
            self.jobs_dir
            / str(job_id)
        )

        if not job_dir.exists():
            return {
                "job_id":
                    str(job_id),

                "selected_candidate_id":
                    None,

                "best_candidate_id":
                    None,

                "candidates":
                    [],
            }

        candidates = self._job_candidates(
            job_dir
        )

        selected_id = self._selected_id(
            job_dir
        )

        best_candidate = next(
            (
                item
                for item in candidates
                if item["is_best"]
            ),
            None,
        )

        return {
            "job_id":
                str(job_id),

            "selected_candidate_id":
                selected_id,

            "best_candidate_id":
                (
                    best_candidate["id"]
                    if best_candidate
                    else None
                ),

            "candidates":
                candidates,
        }

    def list_candidates(
        self,
        job_id=None,
    ):
        return self.get_context(
            job_id
        )["candidates"]

    def select_candidate(
        self,
        candidate_id,
        job_id=None,
    ):
        context = self.get_context(
            job_id
        )

        if not context["job_id"]:
            raise FileNotFoundError(
                "Không tìm thấy NAS search job có candidate."
            )

        candidate = next(
            (
                item
                for item
                in context["candidates"]
                if str(item["id"])
                == str(candidate_id)
            ),
            None,
        )

        if candidate is None:
            raise ValueError(
                f"Không tìm thấy Candidate {candidate_id} "
                f"trong job {context['job_id']}."
            )

        if candidate["status"] not in (
            "completed",
            "feasible",
            "unknown",
        ):
            raise ValueError(
                "Chỉ chọn candidate đã hoàn thành thành công."
            )

        job_dir = (
            self.jobs_dir
            / context["job_id"]
        )

        payload = {
            "job_id":
                context["job_id"],

            "candidate_id":
                candidate["id"],

            "selected_at":
                datetime.now().isoformat(
                    timespec="seconds"
                ),

            "purpose":
                (
                    "Selected architecture for the next "
                    "mixed-precision / deployment stage."
                ),

            "candidate":
                candidate,
        }

        path = (
            job_dir
            / "selected_candidate.json"
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                payload,
                f,
                indent=2,
                ensure_ascii=False,
            )

        return payload
