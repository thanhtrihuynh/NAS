import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def signature(x):
    a = x["architecture"]

    return tuple(
        (
            tuple(a[f"block{i}_kernels"]),
            int(a[f"block{i}_channels"]),
        )
        for i in range(1, 5)
    )


def dominates(a, b):
    no_worse = (
        float(a["macro_f1"]) >= float(b["macro_f1"])
        and int(a["macs"]) <= int(b["macs"])
        and int(a["params"]) <= int(b["params"])
        and float(a["memory_mb"]) <= float(b["memory_mb"])
    )

    strict = (
        float(a["macro_f1"]) > float(b["macro_f1"])
        or int(a["macs"]) < int(b["macs"])
        or int(a["params"]) < int(b["params"])
        or float(a["memory_mb"]) < float(b["memory_mb"])
    )

    return no_worse and strict


def pareto(items):
    return [
        a for i, a in enumerate(items)
        if not any(
            i != j and dominates(b, a)
            for j, b in enumerate(items)
        )
    ]


def classify(x, baseline, tol):
    f1 = float(x["macro_f1"])
    p = int(x["params"])
    m = int(x["macs"])

    bf1 = float(baseline["macro_f1"])
    bp = int(baseline["params"])
    bm = int(baseline["macs"])

    if (
        f1 >= bf1
        and p <= bp
        and m <= bm
        and (f1 > bf1 or p < bp or m < bm)
    ):
        return "DOMINATES_BASELINE", 1

    if f1 > bf1 and (p > bp or m > bm):
        return "HIGHER_ACCURACY_TRADEOFF", 2

    if (
        f1 < bf1
        and f1 >= bf1 - tol
        and (p < bp or m < bm)
    ):
        return "NEAR_BASELINE_EFFICIENCY", 3

    if (
        f1 < bf1 - tol
        and (p < bp or m < bm)
    ):
        return "AGGRESSIVE_EFFICIENCY_TRADEOFF", 4

    if f1 <= bf1 and p >= bp and m >= bm:
        return "DOMINATED_BY_BASELINE", 6

    return "OTHER_TRADEOFF", 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 2026],
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--f1-tolerance",
        type=float,
        default=0.005,
    )
    args = parser.parse_args()

    baseline = json.loads(
        Path(args.baseline).read_text(
            encoding="utf-8"
        )
    )

    root = Path(args.input_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []

    for seed in args.seeds:
        path = (
            root
            / f"seed_{seed}"
            / "all_trained_candidates.json"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Seed {seed} chưa hoàn tất: {path}"
            )

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        for x in data:
            x = dict(x)
            x["_source_seed"] = seed
            rows.append(x)

    grouped = {}

    for x in rows:
        grouped.setdefault(
            signature(x),
            [],
        ).append(x)

    unique = []

    for _, hits in grouped.items():
        representative = max(
            hits,
            key=lambda x: float(x["macro_f1"]),
        )

        f1s = [
            float(x["macro_f1"])
            for x in hits
        ]

        item = dict(representative)
        item["num_times_sampled"] = len(hits)
        item["source_seeds"] = sorted({
            int(x["_source_seed"])
            for x in hits
        })
        item["short_f1_mean_across_hits"] = (
            sum(f1s) / len(f1s)
        )
        unique.append(item)

    front = pareto(unique)
    classified = []

    for x in front:
        cat, priority = classify(
            x,
            baseline,
            args.f1_tolerance,
        )

        item = dict(x)
        item["baseline_category"] = cat
        item["baseline_priority"] = priority
        item["f1_delta_vs_baseline"] = (
            float(x["macro_f1"])
            - float(baseline["macro_f1"])
        )
        item["params_ratio_vs_baseline"] = (
            int(x["params"])
            / int(baseline["params"])
        )
        item["macs_ratio_vs_baseline"] = (
            int(x["macs"])
            / int(baseline["macs"])
        )

        classified.append(item)

    classified.sort(
        key=lambda x: (
            x["baseline_priority"],
            -float(x["macro_f1"]),
            int(x["macs"]),
            int(x["params"]),
        )
    )

    (out / "global_pareto_front.json").write_text(
        json.dumps(
            classified,
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    counts = Counter(
        x["baseline_category"]
        for x in classified
    )

    summary = {
        "seeds": args.seeds,
        "raw_candidate_evaluations": len(rows),
        "unique_architectures": len(unique),
        "global_pareto_size": len(classified),
        "category_counts": dict(counts),
        "baseline": baseline,
        "f1_tolerance": args.f1_tolerance,
        "note": (
            "This is still based on 3-epoch short training. "
            "Full training is required before final selection."
        ),
    }

    (out / "global_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    fields = [
        "id",
        "_source_seed",
        "generation",
        "macro_f1",
        "accuracy",
        "macs",
        "params",
        "memory_mb",
        "baseline_category",
        "f1_delta_vs_baseline",
        "params_ratio_vs_baseline",
        "macs_ratio_vs_baseline",
        "num_times_sampled",
        "source_seeds",
    ]

    with (
        out / "global_pareto_front.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(classified)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
