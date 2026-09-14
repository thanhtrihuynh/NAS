import json
from pathlib import Path

from config import ARTIFACTS_DIR


def main():
    graph_path = (
        ARTIFACTS_DIR
        / "integer_graph.json"
    )

    if not graph_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy:\n{graph_path}\n"
            "Hãy chạy trước:\n"
            "python -m scripts.inspect_integer_graph"
        )

    with Path(
        graph_path
    ).open(
        "r",
        encoding="utf-8"
    ) as f:
        graph = json.load(f)

    important_keywords = [
        "stem",
        "block1",
        "block2",
        "block3",
        "block4",
        "pool1",
        "pool2",
        "pool3",
        "global",
        "gap",
        "gmp",
        "head",
        "class"
    ]

    print()
    print("=" * 110)
    print("INTEGER GRAPH SUMMARY")
    print("=" * 110)

    for item in graph:
        name = item[
            "original_name"
        ]

        lower_name = (
            name.lower()
        )

        if not any(
            keyword in lower_name
            for keyword in important_keywords
        ):
            continue

        layer_type = item[
            "inner_type"
        ]

        inputs = item[
            "input_sources"
        ]

        output_shape = item[
            "output_shape"
        ]

        print(
            f"{name:<32} "
            f"{layer_type:<28} "
            f"<- {str(inputs):<42} "
            f"out={output_shape}"
        )

    print()
    print("=" * 110)
    print("ADD NODES")
    print("=" * 110)

    for item in graph:
        if (
            "add"
            in item[
                "original_name"
            ].lower()
        ):
            print(
                item[
                    "original_name"
                ],
                "<-",
                item[
                    "input_sources"
                ]
            )

    print()
    print("=" * 110)
    print("CONCAT NODES")
    print("=" * 110)

    for item in graph:
        if (
            "concat"
            in item[
                "original_name"
            ].lower()
        ):
            print(
                item[
                    "original_name"
                ],
                "<-",
                item[
                    "input_sources"
                ]
            )

    print()
    print("=" * 110)
    print("POOLING NODES")
    print("=" * 110)

    for item in graph:
        if (
            "pool"
            in item[
                "original_name"
            ].lower()
        ):
            print(
                item[
                    "original_name"
                ],
                "|",
                item[
                    "inner_type"
                ],
                "<-",
                item[
                    "input_sources"
                ]
            )

    print()
    print("=" * 110)
    print("HEAD / OUTPUT")
    print("=" * 110)

    for item in graph:
        name = (
            item[
                "original_name"
            ]
        )

        lower_name = (
            name.lower()
        )

        if (
            "head"
            in lower_name
            or "class"
            in lower_name
            or "global"
            in lower_name
            or "gap"
            in lower_name
            or "gmp"
            in lower_name
        ):
            print(
                name,
                "|",
                item[
                    "inner_type"
                ],
                "<-",
                item[
                    "input_sources"
                ],
                "| out=",
                item[
                    "output_shape"
                ]
            )


if __name__ == "__main__":
    main()