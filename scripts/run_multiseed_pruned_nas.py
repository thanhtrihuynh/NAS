import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument(
        "--runner",
        default="scripts/run_pruned_evolutionary_nas.py",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 2026],
    )
    args = parser.parse_args()

    base = json.loads(
        Path(args.base_config).read_text(
            encoding="utf-8"
        )
    )

    output_root = Path(args.output_root)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for seed in args.seeds:
        run_dir = output_root / f"seed_{seed}"
        done = run_dir / "search_summary.json"

        if done.exists():
            print(
                f"[SKIP] seed={seed}: completed."
            )
            continue

        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(
                f"{run_dir} contains an incomplete run. "
                "Move/delete it before restarting this seed."
            )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cfg = dict(base)
        cfg["seed"] = seed

        config_path = (
            output_root / f"config_seed_{seed}.json"
        )

        config_path.write_text(
            json.dumps(
                cfg,
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )

        cmd = [
            sys.executable,
            args.runner,
            "--config",
            str(config_path),
            "--output",
            str(run_dir),
        ]

        print("\nRUN:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
