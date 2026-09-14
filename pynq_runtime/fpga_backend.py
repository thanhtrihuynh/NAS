import os
import time
from pathlib import Path

import numpy as np


MAGIC = 0x45434731


class FpgaLayerExecutor:
    def __init__(
        self,
        bitstream=None,
        dma_name=None,
    ):
        try:
            from pynq import (
                Overlay,
                allocate,
            )
        except ImportError as exc:
            raise RuntimeError(
                "pynq package không tồn tại. "
                "File này phải chạy trên PYNQ-Z2."
            ) from exc

        self.allocate = allocate

        if bitstream is None:
            bitstream = os.environ.get(
                "ECG_OVERLAY",
                "/home/xilinx/didong/ecg_fpga/"
                "overlay/ecg_accelerator.bit",
            )

        if dma_name is None:
            dma_name = os.environ.get(
                "ECG_DMA_NAME",
                "axi_dma_0",
            )

        bitstream = str(
            Path(bitstream)
        )

        self.overlay = Overlay(
            bitstream
        )

        if not hasattr(
            self.overlay,
            dma_name,
        ):
            names = [
                name
                for name in dir(self.overlay)
                if "dma" in name.lower()
            ]

            raise RuntimeError(
                f"Không tìm thấy DMA '{dma_name}'. "
                f"DMA candidates: {names}"
            )

        self.dma = getattr(
            self.overlay,
            dma_name,
        )

    @staticmethod
    def _u8_words(values):
        values = np.asarray(
            values,
            dtype=np.int8,
        ).reshape(-1)

        return (
            values.astype(np.int16)
            & 0xFF
        ).astype(np.uint32)

    @staticmethod
    def _i32_words(values):
        values = np.asarray(
            values,
            dtype=np.int32,
        ).reshape(-1)

        return values.view(
            np.uint32
        )

    def execute(
        self,
        x,
        weights,
        bias,
        multiplier,
        shift,
        output_bits,
        dilation=1,
        depthwise=False,
        weight_bits=8,
    ):
        x = np.asarray(
            x,
            dtype=np.int8,
        )

        weights = np.asarray(
            weights,
            dtype=np.int8,
        )

        bias = np.asarray(
            bias,
            dtype=np.int32,
        )

        if x.ndim != 2:
            raise ValueError(
                "FPGA input phải [L,Cin]."
            )

        length, cin = x.shape

        if depthwise:
            kernel_size, k_cin, dm = (
                weights.shape
            )

            if k_cin != cin or dm != 1:
                raise ValueError(
                    "Depthwise shape mismatch."
                )

            cout = cin

        else:
            kernel_size, k_cin, cout = (
                weights.shape
            )

            if k_cin != cin:
                raise ValueError(
                    "Conv Cin mismatch."
                )

        if bias.shape != (
            cout,
        ):
            raise ValueError(
                f"Bias shape {bias.shape} "
                f"không khớp Cout={cout}."
            )

        flags = 0
        flags |= 1 << 0  # SAME

        if depthwise:
            flags |= 1 << 1

        if int(weight_bits) == 4:
            flags |= 1 << 2

        if int(output_bits) == 4:
            flags |= 1 << 3

        input_words = self._u8_words(
            x
        )

        weight_words = self._u8_words(
            weights
        )

        bias_words = self._i32_words(
            bias
        )

        header = np.asarray(
            [
                MAGIC,
                length,
                cin,
                cout,
                kernel_size,
                int(dilation),
                flags,
                int(multiplier),
                int(shift),
                input_words.size,
                weight_words.size,
                bias_words.size,
            ],
            dtype=np.uint32,
        )

        packet = np.concatenate(
            [
                header,
                input_words,
                weight_words,
                bias_words,
            ]
        )

        output_count = (
            length
            * cout
        )

        tx = self.allocate(
            shape=(packet.size,),
            dtype=np.uint32,
        )

        rx = self.allocate(
            shape=(output_count,),
            dtype=np.uint32,
        )

        try:
            tx[:] = packet

            t0 = time.perf_counter()

            # Quan trọng: arm S2MM trước MM2S.
            self.dma.recvchannel.transfer(
                rx
            )

            self.dma.sendchannel.transfer(
                tx
            )

            self.dma.sendchannel.wait()
            self.dma.recvchannel.wait()

            elapsed_ms = (
                time.perf_counter()
                - t0
            ) * 1000.0

            output = (
                rx.view(np.int32)
                .reshape(
                    length,
                    cout,
                )
                .astype(np.int8)
            )

            return (
                output,
                elapsed_ms,
            )

        finally:
            tx.freebuffer()
            rx.freebuffer()
