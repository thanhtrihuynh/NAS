import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import wfdb

from config import (
    CACHE_DIR,
    DATASET_DIR,
    LABELS,
    LABEL_TO_ID,
    NUM_CLASSES,
    RANDOM_SEED,
    SEGMENT_LEN,
)


def _find_csv_dir(dataset_root: Path) -> Path:
    matches = []
    for train_csv in dataset_root.rglob("train.csv"):
        folder = train_csv.parent
        if (folder / "val.csv").exists() and (folder / "test.csv").exists():
            matches.append(folder)

    if not matches:
        raise FileNotFoundError(
            f"Không tìm thấy train.csv, val.csv, test.csv trong {dataset_root}"
        )

    matches.sort(key=lambda p: len(p.parts))
    return matches[0]


def _build_record_index(dataset_root: Path):
    index = {}
    for hea in dataset_root.rglob("*.hea"):
        dat = hea.with_suffix(".dat")
        if dat.exists():
            index.setdefault(hea.stem, hea.with_suffix(""))

    if not index:
        raise FileNotFoundError(
            f"Không tìm thấy file MIT-BIH .hea/.dat trong {dataset_root}"
        )

    return index


def _record_id(record_path):
    return Path(str(record_path).replace("\\", "/")).stem


def _fix_length(x, segment_len=SEGMENT_LEN):
    x = np.asarray(x, dtype=np.float32).reshape(-1)

    if len(x) == segment_len:
        return x

    if len(x) > segment_len:
        center = len(x) // 2
        start = max(0, center - segment_len // 2)
        end = start + segment_len

        if end > len(x):
            end = len(x)
            start = end - segment_len

        return x[start:end]

    left = (segment_len - len(x)) // 2
    right = segment_len - len(x) - left
    return np.pad(x, (left, right), mode="constant")


def _zscore(x):
    x = np.asarray(x, dtype=np.float32)
    mean = float(np.mean(x))
    std = float(np.std(x))

    if std < 1e-6:
        std = 1.0

    return ((x - mean) / std).astype(np.float32)


def preprocess_segment(x, segment_len=SEGMENT_LEN):
    return _zscore(_fix_length(x, segment_len))


def _load_split(csv_path: Path, record_index):
    df = pd.read_csv(csv_path)

    required = {"record_path", "start", "end", "labels"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{csv_path} thiếu cột: {sorted(missing)}")

    df = df[df["labels"].isin(LABELS)].reset_index(drop=True)

    record_cache = {}
    X, y, meta = [], [], []
    skipped = 0

    for i, row in df.iterrows():
        try:
            rid = _record_id(row["record_path"])

            if rid not in record_index:
                raise FileNotFoundError(f"Không tìm thấy record {rid}")

            base = str(record_index[rid])

            if base not in record_cache:
                signal, fields = wfdb.rdsamp(base)
                record_cache[base] = (
                    signal.astype(np.float32),
                    list(fields.get("sig_name", [])),
                )

            signal, sig_names = record_cache[base]

            channel_name = str(row.get("channel", "MLII"))
            ch = sig_names.index(channel_name) if channel_name in sig_names else 0

            start = int(row["start"])
            end = int(row["end"])

            segment = signal[start:end + 1, ch]
            segment = preprocess_segment(segment)

            X.append(segment)
            y.append(LABEL_TO_ID[str(row["labels"])])

            meta.append({
                "record_path": str(row["record_path"]),
                "record_id": rid,
                "channel": channel_name,
                "start": start,
                "end": end,
                "label": str(row["labels"]),
            })

        except Exception as exc:
            skipped += 1
            if skipped <= 5:
                warnings.warn(f"Bỏ qua dòng {i} của {csv_path.name}: {exc}")

    if not X:
        raise RuntimeError(f"Không tạo được dữ liệu từ {csv_path}")

    X = np.stack(X).astype(np.float32)[..., np.newaxis]
    y = np.asarray(y, dtype=np.int64)
    meta = pd.DataFrame(meta)

    print(f"{csv_path.name}: X={X.shape}, y={y.shape}, skipped={skipped}")

    return X, y, meta


def _augment_training(X, y, seed=RANDOM_SEED):
    """
    Giữ quy mô augmentation đã dùng trong project:
      V: thêm 1 bản augmented  -> V x2
      A: thêm 4 bản augmented  -> A x5

    Với split cũ:
      raw train = 70,043
      augmented train = 82,358
    """
    rng = np.random.default_rng(seed)
    xs = [X]
    ys = [y]

    def add_class(class_id, repeats):
        mask = y == class_id
        base = X[mask]

        for _ in range(repeats):
            scale = rng.normal(
                1.0, 0.025, size=(len(base), 1, 1)
            ).astype(np.float32)
            noise = rng.normal(
                0.0, 0.012, size=base.shape
            ).astype(np.float32)

            aug = base * scale + noise

            mean = aug.mean(axis=1, keepdims=True)
            std = aug.std(axis=1, keepdims=True)
            std = np.where(std < 1e-6, 1.0, std)

            aug = ((aug - mean) / std).astype(np.float32)

            xs.append(aug)
            ys.append(
                np.full(len(base), class_id, dtype=np.int64)
            )

    add_class(LABEL_TO_ID["V"], 1)
    add_class(LABEL_TO_ID["A"], 4)

    X_aug = np.concatenate(xs, axis=0)
    y_aug = np.concatenate(ys, axis=0)

    order = rng.permutation(len(y_aug))
    return X_aug[order], y_aug[order]


def _cache_paths():
    return (
        CACHE_DIR / "ecg_arrays_v1.npz",
        CACHE_DIR / "ecg_meta_v1.json",
    )


def load_train_val_test(
    dataset_root=DATASET_DIR,
    use_cache=True,
    augment_train=True,
    seed=RANDOM_SEED,
):
    dataset_root = Path(dataset_root)
    arrays_path, meta_path = _cache_paths()

    if use_cache and arrays_path.exists() and meta_path.exists():
        print(f"Load cache: {arrays_path}")
        d = np.load(arrays_path)
        meta_json = json.loads(meta_path.read_text(encoding="utf-8"))

        X_train_raw = d["X_train_raw"].astype(np.float32)
        y_train_raw = d["y_train_raw"].astype(np.int64)
        X_val = d["X_val"].astype(np.float32)
        y_val_int = d["y_val"].astype(np.int64)
        X_test = d["X_test"].astype(np.float32)
        y_test_int = d["y_test"].astype(np.int64)

        train_meta = pd.DataFrame(meta_json["train_meta"])
        val_meta = pd.DataFrame(meta_json["val_meta"])
        test_meta = pd.DataFrame(meta_json["test_meta"])

    else:
        if not dataset_root.exists():
            raise FileNotFoundError(f"Không tồn tại {dataset_root}")

        csv_dir = _find_csv_dir(dataset_root)
        record_index = _build_record_index(dataset_root)

        print("CSV_DIR:", csv_dir)
        print("Records:", len(record_index))

        X_train_raw, y_train_raw, train_meta = _load_split(
            csv_dir / "train.csv", record_index
        )
        X_val, y_val_int, val_meta = _load_split(
            csv_dir / "val.csv", record_index
        )
        X_test, y_test_int, test_meta = _load_split(
            csv_dir / "test.csv", record_index
        )

        if use_cache:
            np.savez_compressed(
                arrays_path,
                X_train_raw=X_train_raw,
                y_train_raw=y_train_raw,
                X_val=X_val,
                y_val=y_val_int,
                X_test=X_test,
                y_test=y_test_int,
            )

            meta_path.write_text(
                json.dumps({
                    "train_meta": train_meta.to_dict(orient="records"),
                    "val_meta": val_meta.to_dict(orient="records"),
                    "test_meta": test_meta.to_dict(orient="records"),
                }, ensure_ascii=True),
                encoding="utf-8",
            )

    if augment_train:
        X_train, y_train_int = _augment_training(
            X_train_raw, y_train_raw, seed
        )
    else:
        X_train, y_train_int = X_train_raw, y_train_raw

    y_train = tf.keras.utils.to_categorical(
        y_train_int, NUM_CLASSES
    ).astype(np.float32)
    y_val = tf.keras.utils.to_categorical(
        y_val_int, NUM_CLASSES
    ).astype(np.float32)
    y_test = tf.keras.utils.to_categorical(
        y_test_int, NUM_CLASSES
    ).astype(np.float32)

    print("\nDATA SUMMARY")
    print("X_train:", X_train.shape)
    print("X_val  :", X_val.shape)
    print("X_test :", X_test.shape)

    return (
        X_train, y_train, y_train_int, train_meta,
        X_val, y_val, y_val_int, val_meta,
        X_test, y_test, y_test_int, test_meta,
    )


def make_sample_weights(y_int):
    counts = np.bincount(
        y_int, minlength=NUM_CLASSES
    ).astype(np.float64)

    total = counts.sum()
    raw = np.zeros(NUM_CLASSES, dtype=np.float64)
    valid = counts > 0
    raw[valid] = total / (NUM_CLASSES * counts[valid])

    smooth = np.sqrt(raw)
    smooth = np.clip(smooth, 0.5, 3.0)

    weights = smooth[y_int].astype(np.float32)
    weights /= np.mean(weights)
    return weights
