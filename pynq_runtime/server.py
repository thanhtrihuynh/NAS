import argparse
import json
import os
from http.server import (
    BaseHTTPRequestHandler,
    ThreadingHTTPServer,
)

from hybrid_model import (
    HybridFpgaRuntime,
)


RUNTIME = None


class Handler(
    BaseHTTPRequestHandler
):
    def _json(
        self,
        status,
        data,
    ):
        body = json.dumps(
            data,
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(
            status
        )

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.end_headers()

        self.wfile.write(
            body
        )

    def do_GET(self):
        if self.path == "/health":
            self._json(
                200,
                {
                    "ok": True,
                    "runtime":
                        "PYNQ_FPGA_HYBRID",
                },
            )

            return

        self._json(
            404,
            {
                "error":
                    "not_found"
            },
        )

    def do_POST(self):
        if self.path != "/api/infer":
            self._json(
                404,
                {
                    "error":
                        "not_found"
                },
            )

            return

        try:
            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

            payload = json.loads(
                self.rfile.read(
                    content_length
                ).decode(
                    "utf-8"
                )
            )

            sample = payload[
                "sample"
            ]

            result = RUNTIME.infer(
                sample[
                    "normalized"
                ]
            )

            result[
                "record_id"
            ] = sample.get(
                "record_id"
            )

            result[
                "true_label"
            ] = sample.get(
                "label"
            )

            result[
                "correct"
            ] = (
                result[
                    "predicted_label"
                ]
                == sample.get(
                    "label"
                )
            )

            self._json(
                200,
                result,
            )

        except Exception as exc:
            self._json(
                500,
                {
                    "error":
                        type(exc).__name__,

                    "detail":
                        str(exc),
                },
            )

    def log_message(
        self,
        fmt,
        *args,
    ):
        print(
            "[HTTP]",
            fmt % args,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        default="0.0.0.0",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=9000,
    )

    parser.add_argument(
        "--deployment",
        default=os.environ.get(
            "ECG_DEPLOYMENT",
            "/home/xilinx/didong/ecg_fpga/"
            "final_w4a4_p99_9",
        ),
    )

    parser.add_argument(
        "--bitstream",
        default=os.environ.get(
            "ECG_OVERLAY",
            "/home/xilinx/didong/ecg_fpga/"
            "overlay/ecg_accelerator.bit",
        ),
    )

    parser.add_argument(
        "--dma",
        default=os.environ.get(
            "ECG_DMA_NAME",
            "axi_dma_0",
        ),
    )

    args = parser.parse_args()

    global RUNTIME

    RUNTIME = HybridFpgaRuntime(
        deployment_root=
            args.deployment,

        bitstream=
            args.bitstream,

        dma_name=
            args.dma,
    )

    server = ThreadingHTTPServer(
        (
            args.host,
            args.port,
        ),
        Handler,
    )

    print(
        f"PYNQ runtime listening on "
        f"http://{args.host}:{args.port}"
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
