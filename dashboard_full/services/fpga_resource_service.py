
import re
from pathlib import Path


class FpgaResourceService:
    def __init__(self, project_root):
        self.root = Path(project_root)

        self.report_candidates = [
            # Prefer project reports.
            self.root
            / "pynq"
            / "overlay"
            / "system_wrapper_utilization_implemented.rpt",

            self.root
            / "pynq"
            / "overlay"
            / "system_wrapper_utilization_routed.rpt",

            self.root
            / "pynq"
            / "overlay"
            / "system_wrapper_utilization_placed.rpt",

            self.root
            / "pynq"
            / "overlay"
            / "system_wrapper_utilization_impl.rpt",

            self.root
            / "system_wrapper_utilization_implemented.rpt",

            self.root
            / "system_wrapper_utilization_routed.rpt",

            self.root
            / "system_wrapper_utilization_placed.rpt",

            # Dashboard fallback report.
            self.root
            / "dashboard_full"
            / "fpga_reports"
            / "system_wrapper_utilization_implemented.rpt",

            self.root
            / "dashboard_full"
            / "fpga_reports"
            / "system_wrapper_utilization_placed.rpt",

            # Synthesis is last and is not trusted if it reports zeros.
            self.root
            / "pynq"
            / "overlay"
            / "system_wrapper_utilization_synth.rpt",

            self.root
            / "dashboard_full"
            / "fpga_reports"
            / "system_wrapper_utilization_synth.rpt",
        ]

    @staticmethod
    def _header(text, key):
        match = re.search(
            rf"^\|\s*{re.escape(key)}\s*:\s*(.*?)\s*$",
            text,
            flags=re.MULTILINE,
        )

        return (
            match.group(1).strip()
            if match
            else None
        )

    @staticmethod
    def _normalize_name(value):
        value = value.strip()

        # Remove trailing report footnote markers such as "*".
        value = re.sub(
            r"\*+$",
            "",
            value,
        )

        return re.sub(
            r"\s+",
            " ",
            value,
        ).strip().lower()

    @classmethod
    def _table_rows(cls, text):
        rows = []

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not (
                line.startswith("|")
                and line.endswith("|")
            ):
                continue

            cells = [
                cell.strip()
                for cell
                in line[1:-1].split("|")
            ]

            if len(cells) < 5:
                continue

            rows.append(cells)

        return rows

    @staticmethod
    def _to_int(value):
        value = str(value).strip()

        if not value:
            return None

        value = value.replace(
            ",",
            "",
        )

        if not re.fullmatch(
            r"[0-9]+",
            value,
        ):
            return None

        return int(value)

    @staticmethod
    def _to_float(value):
        value = str(value).strip()

        if not value:
            return None

        if not re.fullmatch(
            r"[0-9]+(?:\.[0-9]+)?",
            value,
        ):
            return None

        return float(value)

    @classmethod
    def _find_resource(
        cls,
        text,
        aliases,
    ):
        wanted = {
            cls._normalize_name(
                alias
            )
            for alias in aliases
        }

        for cells in cls._table_rows(
            text
        ):
            name = cls._normalize_name(
                cells[0]
            )

            if name not in wanted:
                continue

            # Standard Vivado utilization layout:
            # Site Type | Used | Fixed | Prohibited | Available | Util%
            used = (
                cls._to_int(
                    cells[1]
                )
                if len(cells) > 1
                else None
            )

            available = (
                cls._to_int(
                    cells[4]
                )
                if len(cells) > 4
                else None
            )

            percent = (
                cls._to_float(
                    cells[5]
                )
                if len(cells) > 5
                else None
            )

            if (
                used is not None
                and available is not None
            ):
                if percent is None:
                    percent = (
                        100.0
                        * used
                        / available
                        if available
                        else 0.0
                    )

                return {
                    "used":
                        used,

                    "available":
                        available,

                    "percent":
                        percent,

                    "matched_row":
                        cells[0],
                }

        return None

    @staticmethod
    def _black_box_count(text):
        matches = list(
            re.finditer(
                r"^(?:8|9)\. Black Boxes\s*$",
                text,
                flags=re.MULTILINE,
            )
        )

        if not matches:
            return 0

        start = matches[-1].end()
        tail = text[start:]

        # Stop at the next numbered section.
        end_match = re.search(
            r"^\d+\. Instantiated Netlists\s*$",
            tail,
            flags=re.MULTILINE,
        )

        if end_match:
            tail = tail[
                :end_match.start()
            ]

        counts = re.findall(
            r"^\|\s*[^|]+\|\s*([0-9]+)\s*\|",
            tail,
            flags=re.MULTILINE,
        )

        return sum(
            int(value)
            for value in counts
        )

    def _find_report(self):
        for path in self.report_candidates:
            if path.exists():
                return path

        return None

    def get(self):
        fallback = {
            "lut": {
                "used": None,
                "available": 53200,
                "percent": None,
            },

            "ff": {
                "used": None,
                "available": 106400,
                "percent": None,
            },

            "bram": {
                "used": None,
                "available": 140,
                "percent": None,
            },

            "dsp": {
                "used": None,
                "available": 220,
                "percent": None,
            },
        }

        report = self._find_report()

        if report is None:
            return {
                "board":
                    "PYNQ-Z2",

                "device":
                    "xc7z020clg400-1",

                "clock_mhz":
                    40.0,

                "report_path":
                    None,

                "report_name":
                    None,

                "report_state":
                    None,

                "actual_usage_available":
                    False,

                "reason":
                    "No Vivado utilization report was found.",

                "resources":
                    fallback,
            }

        text = report.read_text(
            encoding="utf-8",
            errors="replace",
        )

        resources = {
            "lut":
                self._find_resource(
                    text,
                    [
                        "Slice LUTs",
                        "Slice LUTs*",
                        "CLB LUTs",
                    ],
                ),

            "ff":
                self._find_resource(
                    text,
                    [
                        "Slice Registers",
                        "CLB Registers",
                        "Register as Flip Flop",
                    ],
                ),

            "bram":
                self._find_resource(
                    text,
                    [
                        "Block RAM Tile",
                        "Block RAM Tiles",
                    ],
                ),

            "dsp":
                self._find_resource(
                    text,
                    [
                        "DSPs",
                        "DSP48E1 only",
                        "DSP48E1",
                    ],
                ),
        }

        for key in resources:
            if resources[key] is None:
                resources[key] = dict(
                    fallback[key]
                )

        design_state = (
            self._header(
                text,
                "Design State",
            )
            or ""
        )

        black_boxes = self._black_box_count(
            text
        )

        positive_values = [
            resource["used"]
            for resource in resources.values()
            if (
                resource["used"]
                is not None
                and resource["used"] > 0
            )
        ]

        is_implemented_stage = (
            "placed"
            in design_state.lower()
            or "routed"
            in design_state.lower()
            or "implemented"
            in design_state.lower()
        )

        actual = (
            is_implemented_stage
            and len(
                positive_values
            ) >= 3
        )

        if actual:
            reason = (
                "FPGA resources parsed from "
                f"{report.name} "
                f"({design_state})."
            )
        else:
            reason = (
                "The available utilization report is not a trustworthy "
                "placed/routed implementation report."
            )

        if not actual:
            # Never show misleading 0% from synthesis/black-box reports.
            for key in resources:
                resources[key] = {
                    "used":
                        None,

                    "available":
                        resources[key].get(
                            "available"
                        )
                        or fallback[key][
                            "available"
                        ],

                    "percent":
                        None,
                }

        return {
            "board":
                "PYNQ-Z2",

            "device":
                (
                    self._header(
                        text,
                        "Device",
                    )
                    or "xc7z020clg400-1"
                ),

            "clock_mhz":
                40.0,

            "report_path":
                str(report),

            "report_name":
                report.name,

            "report_state":
                design_state,

            "black_boxes":
                black_boxes,

            "actual_usage_available":
                actual,

            "reason":
                reason,

            "resources":
                resources,
        }
