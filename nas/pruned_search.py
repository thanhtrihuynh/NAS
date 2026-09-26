import math
from itertools import combinations_with_replacement

from nas.analytical_cost import estimate_architecture_cost


DEFAULT_KERNEL_TRIPLETS = (
    (3, 5, 7),
    (3, 5, 9),
    (3, 7, 9),
    (5, 7, 9),
)
DEFAULT_CHANNEL_CHOICES = tuple(range(4, 65, 4))

# Backward-compatible aliases for other project files that may import them.
KERNEL_TRIPLETS = DEFAULT_KERNEL_TRIPLETS
CHANNEL_CHOICES = DEFAULT_CHANNEL_CHOICES
CHANNEL_TUPLES = tuple(
    combinations_with_replacement(CHANNEL_CHOICES, 4)
)


def normalize_search_space_config(search_space=None):
    """Return a validated, JSON-serializable NAS search-space config.

    Older config files that do not contain ``search_space`` automatically
    receive the original kernel/channel choices, so existing experiments keep
    their previous behavior.
    """
    raw = search_space or {}

    kernels_raw = raw.get(
        "kernel_choices",
        raw.get("kernel_triplets", DEFAULT_KERNEL_TRIPLETS),
    )
    channels_raw = raw.get("channel_choices", DEFAULT_CHANNEL_CHOICES)

    kernel_choices = []
    seen_kernels = set()
    for item in kernels_raw:
        values = tuple(int(x) for x in item)
        if len(values) != 3:
            raise ValueError(
                f"Each kernel choice must contain exactly 3 values: {item}"
            )
        if any(x <= 0 for x in values):
            raise ValueError(f"Kernel sizes must be positive: {item}")
        if values not in seen_kernels:
            kernel_choices.append(values)
            seen_kernels.add(values)

    channel_choices = tuple(
        sorted({int(x) for x in channels_raw})
    )
    if not kernel_choices:
        raise ValueError("At least one kernel choice is required")
    if not channel_choices:
        raise ValueError("At least one channel choice is required")
    if any(x <= 0 for x in channel_choices):
        raise ValueError("Channel choices must be positive integers")

    return {
        "kernel_choices": [list(x) for x in kernel_choices],
        "channel_choices": list(channel_choices),
    }


def _resolved_search_space(search_space=None):
    cfg = normalize_search_space_config(search_space)
    kernels = tuple(tuple(x) for x in cfg["kernel_choices"])
    channels = tuple(cfg["channel_choices"])
    return kernels, channels


def theoretical_search_space_size(search_space=None):
    kernel_choices, channel_choices = _resolved_search_space(search_space)
    # Architecture currently has four Mini-Inception blocks.
    return (
        len(kernel_choices) ** 4
        * math.comb(len(channel_choices) + 4 - 1, 4)
    )


def normalize_architecture(a):
    out = {}
    for i in range(1, 5):
        out[f"block{i}_kernels"] = [
            int(x) for x in a[f"block{i}_kernels"]
        ]
        out[f"block{i}_channels"] = int(
            a[f"block{i}_channels"]
        )
    return out


def architecture_signature(a):
    a = normalize_architecture(a)
    return tuple(
        (
            tuple(a[f"block{i}_kernels"]),
            a[f"block{i}_channels"],
        )
        for i in range(1, 5)
    )


def validate_structure(a, search_space=None):
    a = normalize_architecture(a)
    kernel_choices, channel_choices = _resolved_search_space(search_space)
    channels = []

    for i in range(1, 5):
        kernels = tuple(a[f"block{i}_kernels"])
        channel = a[f"block{i}_channels"]

        if kernels not in kernel_choices:
            return False, f"block{i}: kernel triplet invalid for selected search space"

        if channel not in channel_choices:
            return False, f"block{i}: channel invalid for selected search space"

        channels.append(channel)

    if channels != sorted(channels):
        return False, "C1 <= C2 <= C3 <= C4 violated"

    return True, "OK"


def static_filter(
    a,
    max_params,
    max_macs,
    max_memory_mb,
    search_space=None,
):
    valid, reason = validate_structure(a, search_space)

    if not valid:
        return {
            "passed": False,
            "reasons": [reason],
            "cost": None,
        }

    cost = estimate_architecture_cost(a)
    reasons = []

    if cost.params > int(max_params):
        reasons.append(
            f"params {cost.params} > {int(max_params)}"
        )

    if cost.macs > int(max_macs):
        reasons.append(
            f"macs {cost.macs} > {int(max_macs)}"
        )

    if cost.memory_mb > float(max_memory_mb):
        reasons.append(
            f"memory {cost.memory_mb:.9f} > "
            f"{float(max_memory_mb):.9f}"
        )

    return {
        "passed": not reasons,
        "reasons": reasons,
        "cost": cost.to_dict(),
    }


def random_architecture(rng, search_space=None):
    kernel_choices, channel_choices = _resolved_search_space(search_space)
    channel_tuples = tuple(
        combinations_with_replacement(channel_choices, 4)
    )
    channels = rng.choice(channel_tuples)

    a = {}
    for i in range(1, 5):
        a[f"block{i}_kernels"] = list(
            rng.choice(kernel_choices)
        )
        a[f"block{i}_channels"] = int(channels[i - 1])

    return a


def crossover(a, b, rng, rate):
    if rng.random() >= rate:
        return normalize_architecture(a)

    child = {}
    channels = []

    for i in range(1, 5):
        pk = a if rng.random() < 0.5 else b
        pc = a if rng.random() < 0.5 else b

        child[f"block{i}_kernels"] = list(
            pk[f"block{i}_kernels"]
        )
        channels.append(
            int(pc[f"block{i}_channels"])
        )

    channels.sort()

    for i, c in enumerate(channels, start=1):
        child[f"block{i}_channels"] = c

    return child


def mutate(a, rng, rate, search_space=None):
    child = normalize_architecture(a)
    kernel_choices, channel_choices = _resolved_search_space(search_space)

    channels = [
        child[f"block{i}_channels"]
        for i in range(1, 5)
    ]

    for i in range(1, 5):
        if rng.random() < rate:
            current = tuple(child[f"block{i}_kernels"])
            options = [x for x in kernel_choices if x != current]
            if options:
                child[f"block{i}_kernels"] = list(
                    rng.choice(options)
                )

        if rng.random() < rate:
            channels[i - 1] = rng.choice(channel_choices)

    channels.sort()

    for i, c in enumerate(channels, start=1):
        child[f"block{i}_channels"] = int(c)

    return child

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


def pareto_front(population):
    front = []

    for i, candidate in enumerate(population):
        if not any(
            i != j and dominates(other, candidate)
            for j, other in enumerate(population)
        ):
            front.append(candidate)

    return front


def non_dominated_sort(population):
    remaining = list(population)
    fronts = []
    rank = 0

    while remaining:
        front = pareto_front(remaining)

        for x in front:
            x["_pareto_rank"] = rank

        front_ids = {id(x) for x in front}
        remaining = [
            x for x in remaining
            if id(x) not in front_ids
        ]

        fronts.append(front)
        rank += 1

    return fronts


def crowding_distance(front):
    if not front:
        return {}

    d = {id(x): 0.0 for x in front}

    if len(front) <= 2:
        for x in front:
            d[id(x)] = float("inf")
        return d

    for key in (
        "macro_f1",
        "macs",
        "params",
        "memory_mb",
    ):
        ordered = sorted(
            front,
            key=lambda x: float(x[key]),
        )

        d[id(ordered[0])] = float("inf")
        d[id(ordered[-1])] = float("inf")

        lo = float(ordered[0][key])
        hi = float(ordered[-1][key])

        if hi == lo:
            continue

        for i in range(1, len(ordered) - 1):
            if math.isinf(d[id(ordered[i])]):
                continue

            prev_v = float(ordered[i - 1][key])
            next_v = float(ordered[i + 1][key])

            d[id(ordered[i])] += abs(
                next_v - prev_v
            ) / (hi - lo)

    return d


def assign_rank_and_crowding(population):
    fronts = non_dominated_sort(population)

    for front in fronts:
        dist = crowding_distance(front)
        for x in front:
            x["_crowding"] = dist[id(x)]

    return fronts


def tournament_select(population, rng, size):
    contestants = rng.sample(
        population,
        k=min(int(size), len(population)),
    )

    return min(
        contestants,
        key=lambda x: (
            int(x.get("_pareto_rank", 10**9)),
            -float(x.get("_crowding", 0.0)),
            -float(x["macro_f1"]),
            int(x["macs"]),
        ),
    )


def select_elites(population, count):
    assign_rank_and_crowding(population)

    ordered = sorted(
        population,
        key=lambda x: (
            int(x["_pareto_rank"]),
            -float(x["_crowding"]),
            -float(x["macro_f1"]),
            int(x["macs"]),
            int(x["params"]),
        ),
    )

    return ordered[:int(count)]


def select_final_candidate(front, tolerance):
    best_f1 = max(
        float(x["macro_f1"]) for x in front
    )

    eligible = [
        x for x in front
        if float(x["macro_f1"]) >= best_f1 - float(tolerance)
    ]

    return min(
        eligible,
        key=lambda x: (
            int(x["macs"]),
            float(x["memory_mb"]),
            int(x["params"]),
            -float(x["macro_f1"]),
        ),
    )
