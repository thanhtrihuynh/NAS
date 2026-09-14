import json
import urllib.error
import urllib.request


class PynqService:
    @staticmethod
    def send_sample(
        host,
        port,
        sample,
        execution_mode="hybrid",
        timeout=120.0,
    ):
        url = (
            f"http://{host}:{int(port)}"
            "/api/infer"
        )

        payload = {
            "execution_mode":
                execution_mode,

            "sample": {
                "token":
                    sample.get("token"),

                "center":
                    sample.get("center"),

                "record_id":
                    sample["record_id"],

                "filename":
                    sample["filename"],

                "label":
                    sample["label"],

                "channel":
                    sample["channel"],

                "fs":
                    sample["fs"],

                "shape":
                    sample["shape"],

                "normalized":
                    sample["normalized"],
            },
        }

        body = json.dumps(
            payload
        ).encode("utf-8")

        request = urllib.request.Request(
            url=url,
            data=body,
            headers={
                "Content-Type":
                    "application/json"
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response:
                text = (
                    response
                    .read()
                    .decode("utf-8")
                )

                result = json.loads(
                    text
                )

                return {
                    "success": True,
                    "url": url,
                    "payload_meta": {
                        "record_id":
                            sample["record_id"],

                        "beat_token":
                            sample.get("token"),

                        "center_sample":
                            sample.get("center"),

                        "label":
                            sample["label"],
                    },
                    "response": result,
                }

        except urllib.error.HTTPError as exc:
            try:
                detail = (
                    exc.read()
                    .decode(
                        "utf-8",
                        errors="replace",
                    )
                )
            except Exception:
                detail = str(exc)

            return {
                "success": False,
                "url": url,
                "error": f"HTTP {exc.code}",
                "detail": detail,
                "message": (
                    "PYNQ runtime trả về lỗi khi chạy inference."
                ),
            }

        except urllib.error.URLError as exc:
            return {
                "success": False,
                "url": url,
                "error": str(exc),
                "message": (
                    "Không kết nối được PYNQ runtime tại "
                    f"{host}:{int(port)}. "
                    "Hãy kiểm tra server.py trên PYNQ-Z2."
                ),
            }
