import json
from pathlib import Path
from pprint import pprint


ROOT = Path(
    r"D:\HCMUTE\HK I nam 4\DiDong\ECG_NAS_LOCAL"
)

MANIFEST_FILE = (
    ROOT
    / "artifacts"
    / "final_w4a4_p99_9"
    / "deployment_model"
    / "deployment_manifest.json"
)

SCALES_FILE = (
    ROOT
    / "artifacts"
    / "final_w4a4_p99_9"
    / "deployment_scales.json"
)


with open(
    MANIFEST_FILE,
    "r",
    encoding="utf-8"
) as f:
    manifest = json.load(f)


print("=" * 70)
print("BLOCK1 LAYERS")
print("=" * 70)

for name, cfg in manifest["layers"].items():

    if (
        name.startswith("block1")
        or name == "stem_conv"
    ):

        print()
        print("-" * 70)
        print(name)
        print("-" * 70)

        pprint(cfg)


if SCALES_FILE.exists():

    with open(
        SCALES_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        scales = json.load(f)

    print()
    print("=" * 70)
    print("BLOCK1 / POOL1 SCALES")
    print("=" * 70)

    def walk(obj, path=""):
        if isinstance(obj, dict):

            for key, value in obj.items():

                new_path = (
                    f"{path}.{key}"
                    if path
                    else key
                )

                key_lower = str(key).lower()

                if (
                    "block1" in key_lower
                    or "pool1" in key_lower
                    or "stem" in key_lower
                ):

                    print()
                    print(new_path)

                    if isinstance(
                        value,
                        (dict, list)
                    ):
                        pprint(value)
                    else:
                        print(value)

                walk(
                    value,
                    new_path
                )

        elif isinstance(obj, list):

            for i, value in enumerate(obj):

                walk(
                    value,
                    f"{path}[{i}]"
                )

    walk(scales)