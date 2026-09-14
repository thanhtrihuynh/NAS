from pathlib import Path

import numpy as np

from pynq import Overlay, allocate


MAGIC = 0x45434731


class ECGFPGAAccelerator:

    def __init__(
        self,
        bitstream_path,
        dma_name="axi_dma_0"
    ):
        self.bitstream_path = Path(
            bitstream_path
        )

        print(
            "Loading overlay:"
        )

        print(
            self.bitstream_path
        )

        self.overlay = Overlay(
            str(self.bitstream_path)
        )

        print(
            "\nAvailable IP:"
        )

        for name in (
            self.overlay.ip_dict.keys()
        ):
            print(
                " -",
                name
            )

        if not hasattr(
            self.overlay,
            dma_name
        ):
            raise RuntimeError(
                f"Không tìm thấy DMA: "
                f"{dma_name}"
            )

        self.dma = getattr(
            self.overlay,
            dma_name
        )

        print(
            "\nDMA ready:",
            dma_name
        )


    @staticmethod
    def _output_length(
        input_len,
        kernel_size,
        dilation,
        same
    ):
        if same:
            return int(
                input_len
            )

        return int(
            input_len
            - dilation
            * (
                kernel_size - 1
            )
        )


    @staticmethod
    def _to_u32(
        values
    ):
        values = np.asarray(
            values
        )

        return (
            values.astype(
                np.int32
            )
            .view(
                np.uint32
            )
        )


    def run_layer(
        self,
        x,
        weights,
        bias,

        multiplier,
        shift,

        kernel_size,
        dilation=1,

        same=True,
        depthwise=False,

        int4=False,
        output_int4=False
    ):
        x = np.asarray(
            x,
            dtype=np.int8
        )

        weights = np.asarray(
            weights,
            dtype=np.int8
        )

        bias = np.asarray(
            bias,
            dtype=np.int32
        )


        if x.ndim != 2:
            raise ValueError(
                "x phải có shape "
                "[Length, Cin]"
            )


        input_len = int(
            x.shape[0]
        )

        cin = int(
            x.shape[1]
        )


        if depthwise:

            cout = cin

            if weights.size != (
                kernel_size
                * cin
            ):
                raise ValueError(
                    "Depthwise weight phải có "
                    "K × Cin phần tử."
                )

        else:

            if weights.ndim != 3:
                raise ValueError(
                    "Conv weight phải có shape "
                    "[K, Cin, Cout]"
                )

            if weights.shape[0] != kernel_size:
                raise ValueError(
                    "Kernel size không khớp."
                )

            if weights.shape[1] != cin:
                raise ValueError(
                    "Cin không khớp."
                )

            cout = int(
                weights.shape[2]
            )


        if bias.size != cout:
            raise ValueError(
                "Bias size phải bằng Cout."
            )


        flags = 0

        if same:
            flags |= 1 << 0

        if depthwise:
            flags |= 1 << 1

        if int4:
            flags |= 1 << 2

        if output_int4:
            flags |= 1 << 3


        input_flat = (
            x.reshape(-1)
        )

        weight_flat = (
            weights.reshape(-1)
        )

        bias_flat = (
            bias.reshape(-1)
        )


        header = np.array(
            [
                MAGIC,
                input_len,
                cin,
                cout,
                int(kernel_size),
                int(dilation),
                flags,
                int(multiplier),
                int(shift),
                input_flat.size,
                weight_flat.size,
                bias_flat.size
            ],
            dtype=np.int64
        )


        packet = np.concatenate(
            [
                self._to_u32(
                    header
                ),

                self._to_u32(
                    input_flat
                ),

                self._to_u32(
                    weight_flat
                ),

                self._to_u32(
                    bias_flat
                )
            ]
        ).astype(
            np.uint32
        )


        output_len = (
            self._output_length(
                input_len,
                kernel_size,
                dilation,
                same
            )
        )

        output_count = (
            output_len
            * cout
        )


        tx_buffer = allocate(
            shape=(
                packet.size,
            ),
            dtype=np.uint32
        )

        rx_buffer = allocate(
            shape=(
                output_count,
            ),
            dtype=np.uint32
        )


        tx_buffer[:] = packet


        try:

            # Receive phải arm trước.
            self.dma.recvchannel.transfer(
                rx_buffer
            )

            self.dma.sendchannel.transfer(
                tx_buffer
            )


            self.dma.sendchannel.wait()

            self.dma.recvchannel.wait()


            raw_result = np.array(
                rx_buffer,
                copy=True
            )


        finally:

            tx_buffer.freebuffer()

            rx_buffer.freebuffer()


        signed_result = (
            raw_result.view(
                np.int32
            )
        )


        # Accelerator sign-extend INT8
        # thành AXI word 32-bit.
        result_int8 = (
            signed_result
            .astype(
                np.int8
            )
        )


        return result_int8.reshape(
            output_len,
            cout
        )