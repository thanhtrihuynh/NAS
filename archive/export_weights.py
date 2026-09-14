import tensorflow as tf

from config import BEST_NAS_MODEL_PATH, ARTIFACTS_DIR
from quantization.export import export_integer_weights


def main():
    if not BEST_NAS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {BEST_NAS_MODEL_PATH}."
        )

    model = tf.keras.models.load_model(
        str(BEST_NAS_MODEL_PATH),
        compile=False,
    )

    export_integer_weights(
        model,
        bits=8,
        output_path=ARTIFACTS_DIR / "nas_weights_int8.npz",
    )

    export_integer_weights(
        model,
        bits=4,
        output_path=ARTIFACTS_DIR / "nas_weights_int4.npz",
    )


if __name__ == "__main__":
    main()
