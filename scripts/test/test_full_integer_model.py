import numpy as np

from config import (
    VAL_CSV,
    RECORDS_DIR,
    ARTIFACTS_DIR
)

from data.loader import (
    build_xy,
    to_keras
)

from integer_reference.model import (
    IntegerReferenceModel
)


TEST_SAMPLES = 8


def main():
    print(
        "Loading validation data..."
    )

    X_val, y_val, _ = (
        build_xy(
            VAL_CSV,
            RECORDS_DIR
        )
    )

    X_val, _ = to_keras(
        X_val,
        y_val
    )

    X = (
        X_val[
            :TEST_SAMPLES
        ]
    )

    y = (
        y_val[
            :TEST_SAMPLES
        ]
    )

    deployment_dir = (
        ARTIFACTS_DIR
        / "deployment_model"
    )

    print(
        "Loading Integer Reference Model..."
    )

    model = IntegerReferenceModel(
        deployment_dir
    )

    print(
        "Running integer inference..."
    )

    result = model.forward(
        X,
        return_trace=True
    )

    print()
    print(
        "=" * 60
    )
    print(
        "FULL INTEGER MODEL SMOKE TEST"
    )
    print(
        "=" * 60
    )

    print(
        "Input shape:",
        X.shape
    )

    print(
        "Integer logits shape:",
        result[
            "logits_int"
        ].shape
    )

    print(
        "Logit scale:",
        result[
            "logit_scale"
        ]
    )

    print()
    print(
        "True labels:"
    )

    print(
        y
    )

    print()
    print(
        "Predictions:"
    )

    print(
        result[
            "predictions"
        ]
    )

    print()
    print(
        "Integer logits:"
    )

    print(
        result[
            "logits_int"
        ]
    )

    print()
    print(
        "Trace shapes:"
    )

    for (
        name,
        info
    ) in (
        result[
            "trace"
        ].items()
    ):
        tensor = (
            info[
                "tensor"
            ]
        )

        print(
            f"{name:<24} "
            f"shape="
            f"{tensor.shape} "
            f"scale="
            f"{info['scale']:.10f}"
        )

    print()
    print(
        "FULL INTEGER MODEL TEST DONE"
    )


if __name__ == "__main__":
    main()