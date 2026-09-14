import os
import numpy as np
import wfdb

from config import SEGMENT_LEN

_record_cache = {}


def get_record_id(record_path):
    s = str(record_path).replace("\\", "/")
    base = os.path.basename(s)
    return os.path.splitext(base)[0]


def resolve_record_base(record_path, dataset_dir):
    record_id = get_record_id(record_path)
    candidate = os.path.join(str(dataset_dir), record_id)

    if os.path.exists(candidate + ".hea") and os.path.exists(candidate + ".dat"):
        return candidate

    raise FileNotFoundError(
        f"Không tìm thấy record {record_id}. "
        f"Cần có {candidate}.hea và {candidate}.dat"
    )


def load_record(record_path, dataset_dir):
    base = resolve_record_base(record_path, dataset_dir)

    if base not in _record_cache:
        signal, fields = wfdb.rdsamp(base)
        sig_names = fields.get("sig_name", [])
        _record_cache[base] = (signal.astype(np.float32), sig_names)

    return _record_cache[base]


def clear_record_cache():
    _record_cache.clear()


def fix_length(x, segment_len=SEGMENT_LEN):
    x = np.asarray(x, dtype=np.float32).reshape(-1)

    if len(x) == segment_len:
        return x

    if len(x) > segment_len:
        center = len(x) // 2
        half = segment_len // 2
        start = max(0, center - half)
        return x[start:start + segment_len]

    pad_left = (segment_len - len(x)) // 2
    pad_right = segment_len - len(x) - pad_left
    return np.pad(x, (pad_left, pad_right), mode="constant")


def normalize_segment(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)

    mean = np.mean(x)
    std = np.std(x)

    if std < 1e-6:
        std = 1.0

    x = (x - mean) / std
    return x.astype(np.float32)


def preprocess_segment(x, segment_len=SEGMENT_LEN):
    x = fix_length(x, segment_len)
    x = normalize_segment(x)
    return x.astype(np.float32)
