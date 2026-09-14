import tensorflow as tf

from config import BEST_NAS_MODEL_PATH, VAL_CSV, RECORDS_DIR
from data.loader import build_xy, to_keras
from quantization.fake_quant import build_uniform_quant_model
from training.evaluate import evaluate_model


def main():
    if not BEST_NAS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {BEST_NAS_MODEL_PATH}."
        )

    X_val, y_val, _ = build_xy(VAL_CSV, RECORDS_DIR)
    X_val_keras, y_val_cat = to_keras(X_val, y_val)

    fp32_model = tf.keras.models.load_model(
        str(BEST_NAS_MODEL_PATH),
        compile=False,
    )
    fp32_model.trainable = False

    print("\n=== FP32 ===")
    fp32_result = evaluate_model(
        fp32_model,
        X_val_keras,
        y_val_cat,
        print_report=False,
    )
    print("Accuracy:", fp32_result["accuracy"])
    print("Macro F1:", fp32_result["macro_f1"])

    print("\n=== W8A8 FAKE QUANT ===")
    int8_model = build_uniform_quant_model(fp32_model, bits=8)
    int8_result = evaluate_model(
        int8_model,
        X_val_keras,
        y_val_cat,
        print_report=False,
    )
    print("Accuracy:", int8_result["accuracy"])
    print("Macro F1:", int8_result["macro_f1"])

    print("\n=== W4A4 FAKE QUANT ===")
    int4_model = build_uniform_quant_model(fp32_model, bits=4)
    int4_result = evaluate_model(
        int4_model,
        X_val_keras,
        y_val_cat,
        print_report=False,
    )
    print("Accuracy:", int4_result["accuracy"])
    print("Macro F1:", int4_result["macro_f1"])


if __name__ == "__main__":
    main()
