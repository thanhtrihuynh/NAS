import hashlib
import json
from pathlib import Path

from config import ARTIFACTS_DIR


FINAL_DIR = (
    ARTIFACTS_DIR
    / "final_w4a4_p99_9"
)


def sha256_file(path):
    hasher = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            hasher.update(
                chunk
            )

    return hasher.hexdigest()


def main():
    if not FINAL_DIR.exists():
        raise FileNotFoundError(
            FINAL_DIR
        )

    records = {}

    files = sorted(
        [
            path
            for path
            in FINAL_DIR.rglob("*")
            if path.is_file()
        ]
    )

    print()
    print("=" * 90)
    print("FINAL DEPLOYMENT ARTIFACT HASHES")
    print("=" * 90)

    for path in files:
        relative = path.relative_to(
            FINAL_DIR
        )

        digest = sha256_file(
            path
        )

        records[
            str(relative)
        ] = digest

        print(
            f"{str(relative):<60} "
            f"{digest[:16]}..."
        )

    output = {
        "artifact":
            "final_w4a4_p99_9",

        "files":
            records
    }

    output_path = (
        FINAL_DIR
        / "SHA256SUMS.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        "Saved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()