from pathlib import Path

import numpy as np

from runtime_to_pool1 import (
    RuntimeToPool1
)


ROOT = Path(
    r"D:\HCMUTE\HK I nam 4\DiDong\ECG_NAS_LOCAL"
)

STEM_TEST = (
    ROOT
    / "pynq"
    / "real_stem_test"
)

stem_preact = np.load(
    STEM_TEST
    / "expected_int8.npy"
).astype(
    np.int8
)


print(
    "Stem input:",
    stem_preact.shape
)


runtime = RuntimeToPool1(
    ROOT
)


result = runtime.run(
    stem_preact
)


print()
print(
    "Stem act range:",
    result["stem_act"].min(),
    result["stem_act"].max()
)


for name in [
    "b1",
    "b2",
    "b3"
]:

    x = result[
        name
    ]["act"]

    print(
        name,
        "shape=",
        x.shape,
        "range=",
        int(x.min()),
        "...",
        int(x.max())
    )