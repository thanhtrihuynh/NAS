import numpy as np

from config import ID_TO_LABEL


def add_gaussian_noise(x, noise_std=0.03):
    noise = np.random.normal(0, noise_std, size=x.shape).astype(np.float32)
    return x + noise


def amplitude_scaling(x, scale_range=(0.85, 1.15)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return x * scale


def time_shift(x, max_shift=16):
    shift = np.random.randint(-max_shift, max_shift + 1)

    if shift == 0:
        return x.copy()

    y = np.zeros_like(x)

    if shift > 0:
        y[shift:] = x[:-shift]
    else:
        y[:shift] = x[-shift:]

    return y


def baseline_wander(x, max_amp=0.08, max_freq=2.0):
    n = len(x)
    t = np.linspace(0, 1, n)

    amp = np.random.uniform(0.0, max_amp)
    freq = np.random.uniform(0.2, max_freq)
    phase = np.random.uniform(0, 2 * np.pi)

    baseline = amp * np.sin(2 * np.pi * freq * t + phase)
    return x + baseline.astype(np.float32)


def time_stretch_1d(x, stretch_range=(0.92, 1.08)):
    n = len(x)
    stretch = np.random.uniform(stretch_range[0], stretch_range[1])

    old_idx = np.arange(n)
    new_len = max(8, int(n * stretch))
    new_idx = np.linspace(0, n - 1, new_len)

    y = np.interp(new_idx, old_idx, x)

    if new_len > n:
        start = (new_len - n) // 2
        y = y[start:start + n]
    else:
        pad_left = (n - new_len) // 2
        pad_right = n - new_len - pad_left
        y = np.pad(y, (pad_left, pad_right), mode="constant")

    return y.astype(np.float32)


def normalize_after_aug(x):
    x = x.astype(np.float32)
    mean = np.mean(x)
    std = np.std(x)

    if std < 1e-6:
        std = 1.0

    return ((x - mean) / std).astype(np.float32)


def augment_ecg_beat(x):
    y = x.copy().astype(np.float32)

    if np.random.rand() < 0.70:
        y = add_gaussian_noise(y, noise_std=np.random.uniform(0.01, 0.04))

    if np.random.rand() < 0.70:
        y = amplitude_scaling(y, scale_range=(0.85, 1.15))

    if np.random.rand() < 0.60:
        y = time_shift(y, max_shift=16)

    if np.random.rand() < 0.50:
        y = baseline_wander(y, max_amp=0.06, max_freq=1.5)

    if np.random.rand() < 0.35:
        y = time_stretch_1d(y, stretch_range=(0.94, 1.06))

    return normalize_after_aug(y)


def augment_train_by_class(X, y, target_per_class=None):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    X_aug = [X]
    y_aug = [y]

    unique_labels, counts = np.unique(y, return_counts=True)
    count_dict = dict(zip(unique_labels, counts))

    print("Số mẫu ban đầu:")
    for cls in unique_labels:
        print(ID_TO_LABEL[int(cls)], count_dict[int(cls)])

    if target_per_class is None:
        max_count = max(counts)
        target_per_class = {}

        for cls in unique_labels:
            cls = int(cls)
            label_name = ID_TO_LABEL[cls]

            if label_name == "A":
                target_per_class[cls] = min(max_count, count_dict[cls] * 5)
            elif label_name == "V":
                target_per_class[cls] = min(max_count, count_dict[cls] * 2)
            else:
                target_per_class[cls] = count_dict[cls]

    for cls in unique_labels:
        cls = int(cls)
        current_count = count_dict[cls]
        target_count = int(target_per_class.get(cls, current_count))
        need = max(0, target_count - current_count)

        if need == 0:
            continue

        cls_indices = np.where(y == cls)[0]
        new_X = []
        new_y = []

        for _ in range(need):
            idx = np.random.choice(cls_indices)
            new_X.append(augment_ecg_beat(X[idx]))
            new_y.append(cls)

        X_aug.append(np.stack(new_X).astype(np.float32))
        y_aug.append(np.array(new_y).astype(np.int64))

        print(
            "Augment lớp",
            ID_TO_LABEL[cls],
            "| hiện có:",
            current_count,
            "| sinh thêm:",
            need,
            "| sau augment:",
            current_count + need,
        )

    X_out = np.concatenate(X_aug, axis=0)
    y_out = np.concatenate(y_aug, axis=0)

    perm = np.random.permutation(len(X_out))
    X_out = X_out[perm]
    y_out = y_out[perm]

    return X_out.astype(np.float32), y_out.astype(np.int64)
