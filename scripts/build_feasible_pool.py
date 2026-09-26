import argparse
import gzip
import json
import sys
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nas.analytical_cost import estimate_architecture_cost
from nas.pruned_search import (
    CHANNEL_TUPLES,
    KERNEL_TRIPLETS,
    theoretical_search_space_size,
)


def architectures():
    for channels in CHANNEL_TUPLES:
        for kernels in product(
            KERNEL_TRIPLETS,
            repeat=4,
        ):
            a = {}

            for i in range(4):
                a[f"block{i + 1}_kernels"] = list(kernels[i])
                a[f"block{i + 1}_channels"] = int(channels[i])

            yield a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-feasible", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25000)
    args = parser.parse_args()

    cfg = json.loads(
        Path(args.config).read_text(encoding="utf-8")
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = feasible = rejected = 0
    rp = rm = rmem = rmulti = 0

    mins = {
        "params": None,
        "macs": None,
        "memory": None,
    }
    maxs = {
        "params": None,
        "macs": None,
        "memory": None,
    }

    expected = theoretical_search_space_size()
    pool_path = out / "feasible_architectures.jsonl.gz"

    writer = (
        gzip.open(pool_path, "wt", encoding="utf-8")
        if args.save_feasible
        else None
    )

    try:
        for a in architectures():
            total += 1
            c = estimate_architecture_cost(a)

            reasons = []

            if c.params > int(cfg["max_params"]):
                reasons.append("params")
            if c.macs > int(cfg["max_macs"]):
                reasons.append("macs")
            if c.memory_mb > float(cfg["max_memory_mb"]):
                reasons.append("memory")

            if reasons:
                rejected += 1
                rp += int("params" in reasons)
                rm += int("macs" in reasons)
                rmem += int("memory" in reasons)
                rmulti += int(len(reasons) > 1)

            else:
                feasible += 1

                values = {
                    "params": c.params,
                    "macs": c.macs,
                    "memory": c.memory_mb,
                }

                for k, v in values.items():
                    mins[k] = v if mins[k] is None else min(mins[k], v)
                    maxs[k] = v if maxs[k] is None else max(maxs[k], v)

                if writer:
                    writer.write(
                        json.dumps({
                            "architecture": a,
                            **c.to_dict(),
                        }, ensure_ascii=True) + "\n"
                    )

            if (
                args.progress_every > 0
                and total % args.progress_every == 0
            ):
                print(
                    f"[SCAN] {total:,}/{expected:,} | "
                    f"PASS={feasible:,} | REJECT={rejected:,}"
                )

    finally:
        if writer:
            writer.close()

    if total != expected:
        raise RuntimeError(
            f"Expected {expected}, enumerated {total}"
        )

    summary = {
        "theoretical_total": expected,
        "enumerated_total": total,
        "constraints": {
            "max_params": int(cfg["max_params"]),
            "max_macs": int(cfg["max_macs"]),
            "max_memory_mb": float(cfg["max_memory_mb"]),
        },
        "feasible_count": feasible,
        "rejected_count": rejected,
        "feasible_ratio": feasible / total,
        "rejected_ratio": rejected / total,
        "reject_reason_counts": {
            "params": rp,
            "macs": rm,
            "memory": rmem,
            "multiple_constraints": rmulti,
        },
        "feasible_cost_range": {
            "min_params": mins["params"],
            "max_params": maxs["params"],
            "min_macs": mins["macs"],
            "max_macs": maxs["macs"],
            "min_memory_mb": mins["memory"],
            "max_memory_mb": maxs["memory"],
        },
        "saved_feasible_pool": (
            str(pool_path) if args.save_feasible else None
        ),
        "important_note": (
            "All 992,256 architectures are checked for "
            "Params/MACs/Memory only; F1 is not exhaustively evaluated."
        ),
    }

    path = out / "static_scan_summary.json"
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print("\nEXHAUSTIVE STATIC SCAN COMPLETED")
    print(f"Total    : {total:,}")
    print(f"Feasible : {feasible:,}")
    print(f"Rejected : {rejected:,}")
    print(f"Summary  : {path}")


if __name__ == "__main__":
    main()
