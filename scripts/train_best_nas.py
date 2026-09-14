import tensorflow as tf

from config import BEST_NAS_MODEL_PATH
from data.loader import load_train_val_test
from models.nas_model import build_nas_model
from nas.search_space import BEST_CONFIG, save_config
from nas.cost import estimate_macs
from training.train import compile_model, train_model
from training.evaluate import evaluate_model


def main():
    data = load_train_val_test(augment_train=True)

    model = build_nas_model(BEST_CONFIG)
    compile_model(model)

    model.summary()
    print("Parameters:", f"{model.count_params():,}")
    print("MACs:", f"{estimate_macs(model):,}")

    save_config(BEST_CONFIG)

    train_model(
        model,
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        save_path=BEST_NAS_MODEL_PATH,
        early_stopping_patience=7,
    )

    best_model = tf.keras.models.load_model(
        str(BEST_NAS_MODEL_PATH),
        compile=False,
    )

    print("\n=== BEST NAS FP32 VALIDATION ===")
    evaluate_model(
        best_model,
        data["X_val"],
        data["y_val"],
        print_report=True,
    )


if __name__ == "__main__":
    main()
