import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path


class NasService:
    def __init__(self, project_root):
        self.root = Path(project_root)

        self.jobs_dir = (
            self.root
            / "artifacts"
            / "dashboard_nas_jobs"
        )

        self.jobs_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _now():
        return datetime.now().isoformat(
            timespec="seconds"
        )

    def _status_file(self, job_dir):
        return job_dir / "status.json"

    def _write_status(
        self,
        job_dir,
        status,
        **extra,
    ):
        data = {
            "job_id": job_dir.name,
            "status": status,
            "updated_at": self._now(),
            **extra,
        }

        with open(
            self._status_file(job_dir),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
            )

        return data

    def _monitor(
        self,
        process,
        job_dir,
        stdout_file,
        stderr_file,
    ):
        return_code = process.wait()

        stdout_file.close()
        stderr_file.close()

        if return_code == 0:
            self._write_status(
                job_dir,
                "completed",
                return_code=return_code,
                message="NAS search completed.",
            )
        else:
            self._write_status(
                job_dir,
                "failed",
                return_code=return_code,
                message=(
                    "NAS search failed. "
                    "Xem stdout.log và stderr.log."
                ),
            )

    def start(self, config):
        required = [
            self.root / "data" / "loader.py",
            self.root / "models" / "nas_model.py",
            self.root / "nas" / "cost.py",
            self.root / "training" / "train.py",
        ]

        missing = [
            str(path)
            for path in required
            if not path.exists()
        ]

        dataset_cfg = config.get(
            "dataset",
            {},
        )

        for key in (
            "train_csv",
            "val_csv",
        ):
            value = dataset_cfg.get(
                key
            )

            if value:
                path = Path(value)

                if not path.exists():
                    missing.append(
                        str(path)
                    )

        job_id = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + "_"
            + uuid.uuid4().hex[:8]
        )

        job_dir = (
            self.jobs_dir
            / job_id
        )

        job_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        config_path = (
            job_dir
            / "config.json"
        )

        with open(
            config_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                config,
                f,
                indent=2,
                ensure_ascii=False,
            )

        if missing:
            return self._write_status(
                job_dir,
                "waiting_engine",
                config_file=str(config_path),
                missing_files=missing,
                message=(
                    "NAS preflight failed: thiếu source hoặc train/val split. "
                    "Config đã được lưu nhưng chưa chạy search."
                ),
            )

        command = [
            sys.executable,
            "-m",
            "dashboard_full.nas_runner",
            "--config",
            str(config_path),
            "--output",
            str(job_dir),
        ]

        stdout_file = open(
            job_dir / "stdout.log",
            "w",
            encoding="utf-8",
        )

        stderr_file = open(
            job_dir / "stderr.log",
            "w",
            encoding="utf-8",
        )

        env = os.environ.copy()

        # Windows commonly uses cp1252 for child-process stdout/stderr.
        # Project training code prints Vietnamese text (for example
        # "Số mẫu ban đầu"), so force UTF-8 for the NAS subprocess.
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        process = subprocess.Popen(
            command,
            cwd=str(self.root),
            stdout=stdout_file,
            stderr=stderr_file,
            env=env,
        )

        status = self._write_status(
            job_dir,
            "running",
            pid=process.pid,
            command=command,
            config_file=str(config_path),
            message="NAS search started.",
        )

        thread = threading.Thread(
            target=self._monitor,
            args=(
                process,
                job_dir,
                stdout_file,
                stderr_file,
            ),
            daemon=True,
        )

        thread.start()

        return status

    def get_job(self, job_id):
        job_dir = (
            self.jobs_dir
            / job_id
        )

        status_file = (
            self._status_file(
                job_dir
            )
        )

        if not status_file.exists():
            return None

        with open(
            status_file,
            "r",
            encoding="utf-8",
        ) as f:
            status = json.load(f)

        progress_file = (
            job_dir
            / "progress.json"
        )

        if progress_file.exists():
            try:
                with open(
                    progress_file,
                    "r",
                    encoding="utf-8",
                ) as f:
                    status["progress"] = (
                        json.load(f)
                    )
            except Exception:
                pass

        stdout_file = (
            job_dir
            / "stdout.log"
        )

        stderr_file = (
            job_dir
            / "stderr.log"
        )

        if stdout_file.exists():
            status["stdout_tail"] = (
                stdout_file.read_text(
                    encoding="utf-8",
                    errors="replace",
                )[-4000:]
            )

        if stderr_file.exists():
            status["stderr_tail"] = (
                stderr_file.read_text(
                    encoding="utf-8",
                    errors="replace",
                )[-4000:]
            )

        return status
