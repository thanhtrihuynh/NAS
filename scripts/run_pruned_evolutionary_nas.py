import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import (
    load_train_val_test,
    make_sample_weights,
)
from models.nas_model import build_nas_model
from nas.cost import estimate_macs
from nas.pruned_search import (
    architecture_signature,
    assign_rank_and_crowding,
    crossover,
    mutate,
    pareto_front,
    random_architecture,
    select_elites,
    select_final_candidate,
    static_filter,
    theoretical_search_space_size,
    normalize_search_space_config,
    tournament_select,
)


def save_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def clean(candidate):
    return {
        k: v
        for k, v in candidate.items()
        if not str(k).startswith("_")
    }


def evaluate(
    architecture,
    candidate_id,
    generation,
    cfg,
    X_train,
    y_train,
    sample_weight,
    X_val,
    y_val,
    y_val_int,
    output_dir,
):
    static = static_filter(
        architecture,
        cfg["max_params"],
        cfg["max_macs"],
        cfg["max_memory_mb"],
        cfg.get("search_space"),
    )

    if not static["passed"]:
        return None

    c = static["cost"]

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(
        int(cfg["seed"]) + candidate_id
    )

    model = build_nas_model(architecture)

    actual_params = int(model.count_params())
    actual_macs = int(estimate_macs(model))

    if actual_params != int(c["params"]):
        raise RuntimeError(
            f"Params mismatch ID={candidate_id}: "
            f"{actual_params} != {c['params']}"
        )

    if actual_macs != int(c["macs"]):
        raise RuntimeError(
            f"MACs mismatch ID={candidate_id}: "
            f"{actual_macs} != {c['macs']}"
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(cfg.get("learning_rate", 1e-3))
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.01
        ),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(cfg["early_stop_patience"]),
            restore_best_weights=True,
            mode="min",
            verbose=0,
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        sample_weight=sample_weight,
        epochs=int(cfg["epochs_per_candidate"]),
        batch_size=int(cfg["batch_size"]),
        callbacks=callbacks,
        shuffle=True,
        verbose=0,
    )

    probs = model.predict(
        X_val,
        batch_size=max(256, int(cfg["batch_size"])),
        verbose=0,
    )
    pred = np.argmax(probs, axis=1)

    result = {
        "id": candidate_id,
        "generation": generation,
        "architecture": architecture,
        "macro_f1": float(
            f1_score(
                y_val_int,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(y_val_int, pred)
        ),
        "params": actual_params,
        "macs": actual_macs,
        "memory_mb": float(c["memory_mb"]),
        "peak_activation_elements": int(
            c["peak_activation_elements"]
        ),
        "epochs_ran": len(history.history["loss"]),
    }

    candidate_dir = (
        output_dir / f"candidate_{candidate_id:04d}"
    )
    candidate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        candidate_dir / "config.json",
        architecture,
    )
    save_json(
        candidate_dir / "metrics.json",
        result,
    )

    print(
        f"[DONE] ID={candidate_id:04d} Gen={generation} "
        f"F1={result['macro_f1']:.6f} "
        f"ACC={result['accuracy']:.6f} "
        f"MACs={actual_macs:,} "
        f"Params={actual_params:,} "
        f"Mem={c['memory_mb']:.6f} MB"
    )

    del model
    tf.keras.backend.clear_session()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = json.loads(
        Path(args.config).read_text(encoding="utf-8")
    )
    cfg["search_space"] = normalize_search_space_config(
        cfg.get("search_space")
    )
    search_space = cfg["search_space"]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "search_config.json", cfg)

    rng = random.Random(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    tf.keras.utils.set_random_seed(int(cfg["seed"]))

    (
        X_train, y_train, y_train_int, _,
        X_val, y_val, y_val_int, _,
        X_test, y_test, y_test_int, _,
    ) = load_train_val_test(
        augment_train=True,
        seed=int(cfg["seed"]),
    )

    # Test set is intentionally not used in NAS search.
    del X_test, y_test, y_test_int

    sample_weight = make_sample_weights(
        y_train_int
    )

    pop_size = int(cfg["population_size"])
    generations = int(cfg["generations"])
    elite_count = int(cfg["elite_count"])
    tournament_size = int(cfg["tournament_size"])
    mutation_rate = float(cfg["mutation_rate"])
    crossover_rate = float(cfg["crossover_rate"])
    max_attempts = int(cfg["max_proposal_attempts"])

    seen = set()
    rejected = set()
    all_trained = []
    candidate_id = 0

    def try_candidate(a, generation):
        nonlocal candidate_id

        sig = architecture_signature(a)

        if sig in seen:
            return None

        seen.add(sig)

        sf = static_filter(
            a,
            cfg["max_params"],
            cfg["max_macs"],
            cfg["max_memory_mb"],
            search_space,
        )

        if not sf["passed"]:
            rejected.add(sig)
            return None

        result = evaluate(
            a,
            candidate_id,
            generation,
            cfg,
            X_train,
            y_train,
            sample_weight,
            X_val,
            y_val,
            y_val_int,
            out,
        )

        candidate_id += 1
        all_trained.append(result)
        return result

    population = []
    attempts = 0

    while len(population) < pop_size:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                "Cannot create initial population."
            )

        r = try_candidate(
            random_architecture(rng, search_space),
            generation=0,
        )

        if r is not None:
            population.append(r)

    for generation in range(generations):
        if generation > 0:
            assign_rank_and_crowding(population)
            elites = select_elites(
                population,
                elite_count,
            )

            new_population = list(elites)
            attempts = 0

            while len(new_population) < pop_size:
                attempts += 1

                if attempts > max_attempts:
                    raise RuntimeError(
                        f"Generation {generation}: "
                        "max_proposal_attempts exceeded."
                    )

                pa = tournament_select(
                    population,
                    rng,
                    tournament_size,
                )
                pb = tournament_select(
                    population,
                    rng,
                    tournament_size,
                )

                child = crossover(
                    pa["architecture"],
                    pb["architecture"],
                    rng,
                    crossover_rate,
                )

                child = mutate(
                    child,
                    rng,
                    mutation_rate,
                    search_space,
                )

                r = try_candidate(
                    child,
                    generation,
                )

                if r is not None:
                    new_population.append(r)

            population = new_population

        assign_rank_and_crowding(population)
        front = pareto_front(population)

        save_json(
            out / f"generation_{generation:03d}.json",
            [clean(x) for x in population],
        )

        save_json(
            out / f"generation_{generation:03d}_pareto.json",
            [clean(x) for x in front],
        )

        summary = {
            "generation": generation,
            "population_size": len(population),
            "pareto_size": len(front),
            "unique_trained_so_far": len(all_trained),
            "unique_static_rejects": len(rejected),
            "best_macro_f1": max(
                float(x["macro_f1"])
                for x in population
            ),
        }

        save_json(
            out / f"generation_{generation:03d}_summary.json",
            summary,
        )

        print(f"Population: {len(population)}")
        print(f"Pareto size: {len(front)}")
        print(
            f"Unique trained so far: {len(all_trained)}"
        )
        print(
            f"Unique static rejects: {len(rejected)}"
        )

    global_front = pareto_front(all_trained)

    best_f1 = max(
        all_trained,
        key=lambda x: float(x["macro_f1"]),
    )

    selected = select_final_candidate(
        global_front,
        cfg["f1_tolerance"],
    )

    save_json(
        out / "all_trained_candidates.json",
        [clean(x) for x in all_trained],
    )

    save_json(
        out / "pareto_front.json",
        [clean(x) for x in global_front],
    )

    save_json(
        out / "best_candidate.json",
        clean(selected),
    )

    summary = {
        "search_space": theoretical_search_space_size(search_space),
        "unique_trained": len(all_trained),
        "unique_static_rejects": len(rejected),
        "unique_proposals_seen": len(seen),
        "pareto_front_size": len(global_front),
        "best_f1_candidate": int(best_f1["id"]),
        "best_f1_observed": float(best_f1["macro_f1"]),
        "selected_candidate": int(selected["id"]),
        "selected_f1": float(selected["macro_f1"]),
        "selected_macs": int(selected["macs"]),
        "selected_params": int(selected["params"]),
        "selected_memory_mb": float(
            selected["memory_mb"]
        ),
        "test_set_used_for_search": False,
    }

    save_json(
        out / "search_summary.json",
        summary,
    )

    print("\nPRUNED EVOLUTIONARY NAS COMPLETED")
    print(
        f"Search space       : "
        f"{summary['search_space']:,}"
    )
    print(
        f"Unique trained     : "
        f"{summary['unique_trained']}"
    )
    print(
        f"Static rejects     : "
        f"{summary['unique_static_rejects']}"
    )
    print(
        f"Pareto front       : "
        f"{summary['pareto_front_size']}"
    )
    print(
        f"Best F1 observed   : "
        f"{summary['best_f1_observed']:.6f}"
    )
    print(
        f"Selected candidate : "
        f"{summary['selected_candidate']}"
    )


if __name__ == "__main__":
    main()
