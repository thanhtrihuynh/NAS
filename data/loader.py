import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    LABELS,
    LABEL_TO_ID,
    NUM_CLASSES,
    SEGMENT_LEN,
    RECORDS_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
)
from data.preprocessing import (
    get_record_id,
    load_record,
    preprocess_segment,
)
from data.augmentation import augment_train_by_class


def build_xy(csv_path, dataset_dir=RECORDS_DIR, segment_len=SEGMENT_LEN):
    df = pd.read_csv(csv_path)
    df = df[df["labels"].isin(LABELS)].reset_index(drop=True)

    X = []
    y = []
    meta = []
    skipped = 0

    for i, row in df.iterrows():
        try:
            signal, sig_names = load_record(row["record_path"], dataset_dir)

            channel_name = str(row.get("channel", "MLII"))
            ch = sig_names.index(channel_name) if channel_name in sig_names else 0

            start = int(row["start"])
            end = int(row["end"])

            segment = signal[start:end + 1, ch]
            segment = preprocess_segment(segment, segment_len)

            X.append(segment)
            y.append(LABEL_TO_ID[row["labels"]])

            meta.append(
                {
                    "record_path": row["record_path"],
                    "record_id": get_record_id(row["record_path"]),
                    "channel": channel_name,
                    "start": start,
                    "end": end,
                    "label": row["labels"],
                }
            )

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                warnings.warn(f"Bỏ qua dòng {i}: {e}")

    if len(X) == 0:
        raise RuntimeError(
            "Không tạo được dữ liệu. Kiểm tra CSV và thư mục MIT-BIH .dat/.hea."
        )

    X = np.stack(X).astype(np.float32)
    y = np.array(y).astype(np.int64)
    meta_df = pd.DataFrame(meta)

    print(
        os.path.basename(str(csv_path)),
        "X =",
        X.shape,
        "y =",
        y.shape,
        "skipped =",
        skipped,
    )

    return X, y, meta_df


def to_keras(X, y):
    X_keras = X[..., np.newaxis].astype(np.float32)
    y_cat = tf.keras.utils.to_categorical(y, num_classes=NUM_CLASSES)
    return X_keras, y_cat


def load_train_val_test(augment_train=True):
    X_train, y_train, train_meta = build_xy(TRAIN_CSV)
    X_val, y_val, val_meta = build_xy(VAL_CSV)
    X_test, y_test, test_meta = build_xy(TEST_CSV)

    if augment_train:
        X_train, y_train = augment_train_by_class(X_train, y_train)

    X_train_keras, y_train_cat = to_keras(X_train, y_train)
    X_val_keras, y_val_cat = to_keras(X_val, y_val)
    X_test_keras, y_test_cat = to_keras(X_test, y_test)

    return {
        "X_train": X_train_keras,
        "y_train": y_train_cat,
        "y_train_int": y_train,
        "train_meta": train_meta,
        "X_val": X_val_keras,
        "y_val": y_val_cat,
        "y_val_int": y_val,
        "val_meta": val_meta,
        "X_test": X_test_keras,
        "y_test": y_test_cat,
        "y_test_int": y_test,
        "test_meta": test_meta,
    }
