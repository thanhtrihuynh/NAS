from data.loader import load_train_val_test
from nas.search import run_random_search
from nas.search_space import save_config


def main():
    data = load_train_val_test(augment_train=True)

    results = run_random_search(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        num_candidates=20,
        search_epochs=5,
    )

    best = results[0]

    print("\nBEST ARCHITECTURE")
    print("Score:", best["score"])
    print("Macro F1:", best["macro_f1"])
    print("Parameters:", best["params"])
    print("MACs:", best["macs"])
    print("Configuration:")
    print(best["config"])

    save_config(best["config"])


if __name__ == "__main__":
    main()
