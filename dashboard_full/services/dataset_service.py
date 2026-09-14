from pathlib import Path

import math

import numpy as np
import pandas as pd
import wfdb


SEGMENT_LEN = 320
SUPPORTED_LABELS = ["N", "L", "R", "V", "A"]


class DatasetService:
    def __init__(self, project_root):
        self.root = Path(project_root)
        self.datasets_dir = self.root / "datasets"

        if not self.datasets_dir.exists():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục datasets: {self.datasets_dir}"
            )

        self.test_csv = self._detect_test_csv()
        self.df = self._load_test_csv()

        # Quan trọng:
        # record gốc của project hiện nằm trực tiếp trong datasets/.
        # Ví dụ datasets/100.dat, datasets/100.hea, datasets/100.atr.
        self._record_bases = self._scan_record_bases()

        # Cache tối đa vài record đã mở để kéo thanh ECG không phải
        # đọc lại file .dat từ ổ đĩa ở mỗi lần thay đổi vị trí.
        self._signal_cache = {}
        self._signal_cache_order = []
        self._signal_cache_limit = 2

    def _detect_test_csv(self):
        candidates = [
            self.datasets_dir / "splits" / "test.csv",
            self.datasets_dir / "test.csv",
        ]

        for path in candidates:
            if path.exists():
                return path

        return None

    def _load_test_csv(self):
        if self.test_csv is None:
            self.label_col = None
            return pd.DataFrame()

        df = pd.read_csv(self.test_csv).reset_index(drop=True)

        if "labels" in df.columns:
            self.label_col = "labels"
        elif "label" in df.columns:
            self.label_col = "label"
        else:
            self.label_col = None

        if "record_path" in df.columns:
            df["_record_id"] = df["record_path"].map(self._record_id_from_text)
        elif "record" in df.columns:
            df["_record_id"] = df["record"].map(self._record_id_from_text)
        elif "record_id" in df.columns:
            df["_record_id"] = df["record_id"].map(self._record_id_from_text)
        else:
            df["_record_id"] = ""

        if self.label_col is not None:
            df = df[
                df[self.label_col].astype(str).isin(SUPPORTED_LABELS)
            ].reset_index(drop=True)

        return df

    @staticmethod
    def _record_id_from_text(value):
        text = str(value).replace("\\", "/").strip()

        # Hỗ trợ:
        # 100
        # 100.dat
        # datasets/100
        # datasets/100.dat
        # D:/.../datasets/100.dat
        return Path(text).stem

    def _scan_record_bases(self):
        records = {}

        # 1) ƯU TIÊN đúng cấu trúc thực tế của người dùng:
        #    datasets/100.hea + datasets/100.dat
        for hea in sorted(self.datasets_dir.glob("*.hea")):
            base = hea.with_suffix("")
            dat = base.with_suffix(".dat")

            if dat.exists():
                records[base.name] = base

        # 2) Fallback: chỉ bổ sung record chưa tồn tại nếu project khác
        #    để tránh datasets/mitbih/100 ghi đè datasets/100.
        for hea in sorted(self.datasets_dir.rglob("*.hea")):
            if hea.parent == self.datasets_dir:
                continue

            base = hea.with_suffix("")
            dat = base.with_suffix(".dat")

            if dat.exists() and base.name not in records:
                records[base.name] = base

        return dict(
            sorted(
                records.items(),
                key=lambda item: (
                    int(item[0]) if item[0].isdigit() else 10**9,
                    item[0],
                ),
            )
        )

    def refresh(self):
        self.test_csv = self._detect_test_csv()
        self.df = self._load_test_csv()
        self._record_bases = self._scan_record_bases()

        self._signal_cache.clear()
        self._signal_cache_order.clear()

    def _resolve_record(self, record_id):
        record_id = str(record_id).strip()

        if record_id in self._record_bases:
            return self._record_bases[record_id]

        self.refresh()

        if record_id in self._record_bases:
            return self._record_bases[record_id]

        raise FileNotFoundError(
            f"Không tìm thấy record {record_id}. "
            f"Đã tìm trong {self.datasets_dir}"
        )


    def _read_record(self, record_id):
        record_id = str(record_id)

        if record_id in self._signal_cache:
            return self._signal_cache[record_id]

        record_base = self._resolve_record(record_id)

        signal, fields = wfdb.rdsamp(
            str(record_base)
        )

        signal = np.asarray(
            signal,
            dtype=np.float32,
        )

        item = (
            signal,
            fields,
            record_base,
        )

        self._signal_cache[record_id] = item
        self._signal_cache_order.append(record_id)

        while (
            len(self._signal_cache_order)
            > self._signal_cache_limit
        ):
            oldest = self._signal_cache_order.pop(0)

            if oldest != record_id:
                self._signal_cache.pop(
                    oldest,
                    None,
                )

        return item

    def _annotation_path(self, record_id):
        base = self._resolve_record(record_id)
        atr = base.with_suffix(".atr")
        return atr if atr.exists() else None

    @staticmethod
    def _fix_length(x):
        x = np.asarray(x, dtype=np.float32).reshape(-1)

        if len(x) == SEGMENT_LEN:
            return x

        if len(x) > SEGMENT_LEN:
            center = len(x) // 2
            start = center - SEGMENT_LEN // 2
            return x[start:start + SEGMENT_LEN]

        total_pad = SEGMENT_LEN - len(x)
        left = total_pad // 2
        right = total_pad - left

        return np.pad(
            x,
            (left, right),
            mode="constant",
        )

    @staticmethod
    def _zscore(x):
        x = np.asarray(x, dtype=np.float32)

        mean = float(np.mean(x))
        std = float(np.std(x))

        if std < 1e-6:
            std = 1.0

        return ((x - mean) / std).astype(np.float32)

    def labels(self):
        return list(SUPPORTED_LABELS)

    def dataset_status(self):
        root_records = [
            record_id
            for record_id, base in self._record_bases.items()
            if base.parent == self.datasets_dir
        ]

        return {
            "datasets_dir": str(self.datasets_dir),
            "test_csv": str(self.test_csv) if self.test_csv else None,
            "records_found": len(self._record_bases),
            "root_records_found": len(root_records),
            "test_rows": int(len(self.df)),
            "first_records": list(self._record_bases.keys())[:10],
            "test_columns": list(self.df.columns) if not self.df.empty else [],
        }

    def _test_rows_for_record(self, record_id, label=None):
        if self.df.empty or "_record_id" not in self.df.columns:
            return self.df.iloc[0:0]

        rows = self.df[
            self.df["_record_id"].astype(str) == str(record_id)
        ]

        if label and self.label_col is not None:
            rows = rows[
                rows[self.label_col].astype(str) == str(label)
            ]

        return rows

    def _atr_beats(self, record_id, label=None):
        base = self._resolve_record(record_id)

        atr_path = base.with_suffix(".atr")

        if not atr_path.exists():
            return []

        ann = wfdb.rdann(str(base), "atr")

        beats = []

        for ann_index, (sample, symbol) in enumerate(
            zip(ann.sample, ann.symbol)
        ):
            symbol = str(symbol)

            # Model hiện tại của project học đúng 5 class này.
            if symbol not in SUPPORTED_LABELS:
                continue

            if label and symbol != str(label):
                continue

            center = int(sample)

            beats.append(
                {
                    "token": f"atr:{record_id}:{ann_index}",
                    "source": "atr",
                    "record_id": str(record_id),
                    "label": symbol,
                    "channel": None,
                    "start": center - SEGMENT_LEN // 2,
                    "end": center + SEGMENT_LEN // 2 - 1,
                    "center": center,
                    "annotation_index": int(ann_index),
                }
            )

        return beats

    def records(self, label=None, query=None):
        # Record list đọc trực tiếp từ datasets/*.dat + *.hea.
        # Không cần test.csv để record xuất hiện.
        result = []

        for record_id, base in self._record_bases.items():
            if query:
                q = str(query).strip().lower()

                if (
                    q not in record_id.lower()
                    and q not in f"{record_id}.dat".lower()
                ):
                    continue

            # Nếu đang filter class, chỉ hiện record thật sự có class đó
            # trong .atr hoặc test.csv.
            if label:
                atr_count = len(
                    self._atr_beats(
                        record_id,
                        label=label,
                    )
                )

                test_count = len(
                    self._test_rows_for_record(
                        record_id,
                        label=label,
                    )
                )

                if atr_count == 0 and test_count == 0:
                    continue
            else:
                atr_count = None

            header = wfdb.rdheader(str(base))

            channels = list(header.sig_name or [])
            fs = float(header.fs)
            sig_len = int(header.sig_len)

            test_rows = self._test_rows_for_record(record_id)

            label_counts = {}

            if not test_rows.empty and self.label_col is not None:
                label_counts = (
                    test_rows[self.label_col]
                    .astype(str)
                    .value_counts()
                    .to_dict()
                )

            result.append(
                {
                    "record_id": record_id,
                    "filename": f"{record_id}.dat",
                    "record_base": str(base),
                    "direct_dataset_record": base.parent == self.datasets_dir,
                    "has_atr": base.with_suffix(".atr").exists(),
                    "test_segments": int(len(test_rows)),
                    "filtered_atr_count": atr_count,
                    "label_counts": {
                        str(k): int(v)
                        for k, v in label_counts.items()
                    },
                    "channels": channels,
                    "fs": fs,
                    "sig_len": sig_len,
                    "available": True,
                }
            )

        return result

    def _test_row_to_beat(self, row_index, row):
        # Hỗ trợ nhiều kiểu split CSV.
        # Chuẩn project hiện tại kỳ vọng start/end.
        if "start" in row and "end" in row:
            start = int(row["start"])
            end = int(row["end"])
            center = (start + end) // 2

        elif "center" in row:
            center = int(row["center"])
            start = center - SEGMENT_LEN // 2
            end = center + SEGMENT_LEN // 2 - 1

        elif "sample" in row:
            center = int(row["sample"])
            start = center - SEGMENT_LEN // 2
            end = center + SEGMENT_LEN // 2 - 1

        else:
            raise ValueError(
                "test.csv không có start/end, center hoặc sample. "
                f"Các cột hiện tại: {list(self.df.columns)}"
            )

        beat_label = (
            str(row[self.label_col])
            if self.label_col is not None
            else "?"
        )

        return {
            "token": f"test:{int(row_index)}",
            "source": "test",
            "record_id": str(row["_record_id"]),
            "label": beat_label,
            "channel": str(row.get("channel", "MLII")),
            "start": int(start),
            "end": int(end),
            "center": int(center),
            "sample_index": int(row_index),
        }

    def record_beats(self, record_id, source="atr", label=None):
        source = str(source or "atr").lower()

        if source == "atr":
            return self._atr_beats(
                record_id,
                label=label,
            )

        if source != "test":
            raise ValueError(
                "Beat source chỉ hỗ trợ 'atr' hoặc 'test'."
            )

        rows = self._test_rows_for_record(
            record_id,
            label=label,
        )

        beats = []

        for row_index, row in rows.iterrows():
            beats.append(
                self._test_row_to_beat(
                    row_index,
                    row,
                )
            )

        beats.sort(
            key=lambda item: (
                item["center"],
                item["token"],
            )
        )

        return beats

    def record_waveform(self, record_id, channel=None, max_points=5000):
        signal, fields, record_base = self._read_record(
            record_id
        )

        sig_names = list(fields.get("sig_name", []))

        if not sig_names:
            sig_names = [
                f"CH{i}"
                for i in range(signal.shape[1])
            ]

        if channel in sig_names:
            channel_index = sig_names.index(channel)
        else:
            channel_index = 0
            channel = sig_names[0]

        fs = float(fields.get("fs", 360.0))
        full = signal[:, channel_index]
        n = len(full)

        if n <= max_points:
            indices = np.arange(n, dtype=np.int64)
        else:
            indices = np.linspace(
                0,
                n - 1,
                max_points,
                dtype=np.int64,
            )

        # Marker waveform luôn lấy từ .atr vì đây là ground-truth annotation
        # trực tiếp của record MIT-BIH.
        markers = [
            {
                "token": beat["token"],
                "sample": beat["center"],
                "label": beat["label"],
            }
            for beat in self._atr_beats(record_id)
        ]

        return {
            "record_id": str(record_id),
            "filename": f"{record_id}.dat",
            "channel": channel,
            "channels": sig_names,
            "fs": fs,
            "signal_length": int(n),
            "duration_seconds": float(n / fs),
            "sample_indices": indices.astype(int).tolist(),
            "values": full[indices].astype(float).tolist(),
            "markers": markers,
        }


    def record_window(
        self,
        record_id,
        channel=None,
        start_sec=0.0,
        duration_sec=10.0,
    ):
        signal, fields, record_base = self._read_record(
            record_id
        )

        sig_names = list(
            fields.get(
                "sig_name",
                [],
            )
        )

        if not sig_names:
            sig_names = [
                f"CH{i}"
                for i in range(
                    signal.shape[1]
                )
            ]

        if channel in sig_names:
            channel_index = sig_names.index(
                channel
            )
        else:
            channel_index = 0
            channel = sig_names[0]

        fs = float(
            fields.get(
                "fs",
                360.0,
            )
        )

        signal_length = int(
            signal.shape[0]
        )

        total_duration = (
            signal_length
            / fs
        )

        try:
            duration_sec = float(
                duration_sec
            )
        except (TypeError, ValueError):
            duration_sec = 10.0

        if not math.isfinite(
            duration_sec
        ):
            duration_sec = 10.0

        duration_sec = float(
            max(
                1.0,
                min(
                    duration_sec,
                    min(
                        60.0,
                        total_duration,
                    ),
                ),
            )
        )

        max_start = max(
            0.0,
            total_duration - duration_sec,
        )

        try:
            start_sec = float(
                start_sec
            )
        except (TypeError, ValueError):
            start_sec = 0.0

        if not math.isfinite(
            start_sec
        ):
            start_sec = 0.0

        start_sec = float(
            min(
                max(
                    0.0,
                    start_sec,
                ),
                max_start,
            )
        )

        start_sample = int(
            round(
                start_sec
                * fs
            )
        )

        window_samples = max(
            1,
            int(
                round(
                    duration_sec
                    * fs
                )
            ),
        )

        end_sample = min(
            signal_length,
            start_sample
            + window_samples,
        )

        values = signal[
            start_sample:end_sample,
            channel_index,
        ]

        all_beats = self._atr_beats(
            record_id
        )

        markers = [
            {
                "token":
                    beat["token"],

                "sample":
                    int(
                        beat["center"]
                    ),

                "label":
                    beat["label"],

                "relative_sec":
                    float(
                        (
                            beat["center"]
                            - start_sample
                        )
                        / fs
                    ),
            }
            for beat in all_beats
            if (
                start_sample
                <= beat["center"]
                < end_sample
            )
        ]

        return {
            "record_id":
                str(record_id),

            "filename":
                f"{record_id}.dat",

            "channel":
                channel,

            "fs":
                fs,

            "signal_length":
                signal_length,

            "total_duration_seconds":
                float(
                    total_duration
                ),

            "start_sec":
                float(
                    start_sample
                    / fs
                ),

            "end_sec":
                float(
                    end_sample
                    / fs
                ),

            "duration_sec":
                float(
                    (
                        end_sample
                        - start_sample
                    )
                    / fs
                ),

            "start_sample":
                int(
                    start_sample
                ),

            "end_sample":
                int(
                    end_sample
                ),

            "values":
                values.astype(
                    float
                ).tolist(),

            "markers":
                markers,
        }

    def _reference_from_test(self, row_index):
        row_index = int(row_index)

        if self.df.empty:
            raise RuntimeError(
                "Không có test.csv."
            )

        if row_index not in self.df.index:
            raise IndexError(
                f"Test row index không hợp lệ: {row_index}"
            )

        row = self.df.loc[row_index]
        return self._test_row_to_beat(row_index, row)

    def _reference_from_atr(self, record_id, annotation_index):
        record_id = str(record_id)
        annotation_index = int(annotation_index)

        base = self._resolve_record(record_id)
        ann = wfdb.rdann(str(base), "atr")

        if (
            annotation_index < 0
            or annotation_index >= len(ann.sample)
        ):
            raise IndexError(
                f"Annotation index không hợp lệ: {annotation_index}"
            )

        center = int(ann.sample[annotation_index])
        label = str(ann.symbol[annotation_index])

        if label not in SUPPORTED_LABELS:
            raise ValueError(
                f"Annotation {label} không thuộc N/L/R/V/A."
            )

        return {
            "token": f"atr:{record_id}:{annotation_index}",
            "source": "atr",
            "record_id": record_id,
            "label": label,
            "channel": None,
            "start": center - SEGMENT_LEN // 2,
            "end": center + SEGMENT_LEN // 2 - 1,
            "center": center,
            "annotation_index": annotation_index,
        }

    def load_beat(self, token):
        parts = str(token).split(":")

        if parts[0] == "test" and len(parts) == 2:
            ref = self._reference_from_test(parts[1])

        elif parts[0] == "atr" and len(parts) == 3:
            ref = self._reference_from_atr(
                parts[1],
                parts[2],
            )

        else:
            raise ValueError(
                f"Beat token không hợp lệ: {token}"
            )

        record_id = ref["record_id"]
        record_base = self._resolve_record(record_id)

        signal, fields, record_base = self._read_record(
            record_id
        )

        signal = np.asarray(
            signal,
            dtype=np.float32,
        )

        sig_names = list(fields.get("sig_name", []))

        requested_channel = ref.get("channel")

        if requested_channel and requested_channel in sig_names:
            channel_index = sig_names.index(requested_channel)
            channel_name = requested_channel
        else:
            channel_index = 0
            channel_name = (
                str(sig_names[0])
                if sig_names
                else "CH0"
            )

        start = int(ref["start"])
        end = int(ref["end"])
        center = int(ref["center"])

        # Đọc vùng hợp lệ rồi pad nếu heartbeat nằm sát đầu/cuối record.
        valid_start = max(0, start)
        valid_end = min(
            signal.shape[0] - 1,
            end,
        )

        raw = signal[
            valid_start:valid_end + 1,
            channel_index,
        ].astype(np.float32)

        fixed = self._fix_length(raw)
        normalized = self._zscore(fixed)

        fs = float(fields.get("fs", 360.0))

        time_axis = (
            np.arange(
                SEGMENT_LEN,
                dtype=np.float32,
            )
            / fs
        )

        return {
            "token": ref["token"],
            "source": ref["source"],
            "label": ref["label"],
            "record_id": record_id,
            "filename": f"{record_id}.dat",
            "record_local": str(record_base),
            "channel": channel_name,
            "start": start,
            "end": end,
            "center": center,
            "fs": fs,
            "shape": [SEGMENT_LEN, 1],
            "time": time_axis.tolist(),
            "raw": fixed.astype(float).tolist(),
            "normalized": normalized.astype(float).tolist(),
        }

    # Compatibility cho code cũ.
    def load_sample(self, index):
        return self.load_beat(
            f"test:{int(index)}"
        )
