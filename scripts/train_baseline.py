import tensorflow as tf

from config import BASELINE_MODEL_PATH
from data.loader import load_train_val_test
from models.baseline_model import build_baseline_model
from training.train import compile_model, train_model
from training.evaluate import evaluate_model


def main():
    data = load_train_val_test(augment_train=True)

    model = build_baseline_model()
    compile_model(model)

    model.summary()
    print("Parameters:", f"{model.count_params():,}")

    train_model(
        model,
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        save_path=BASELINE_MODEL_PATH,
        early_stopping_patience=9,
    )

    best_model = tf.keras.models.load_model(
        str(BASELINE_MODEL_PATH),
        compile=False,
    )

    print("\n=== BASELINE TEST ===")
    evaluate_model(
        best_model,
        data["X_test"],
        data["y_test"],
        print_report=True,
    )


if __name__ == "__main__":
    main()
