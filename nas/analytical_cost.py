from dataclasses import dataclass

from config import NUM_CLASSES, SEGMENT_LEN


@dataclass(frozen=True)
class ArchitectureCost:
    params: int
    macs: int
    memory_mb: float
    peak_activation_elements: int

    def to_dict(self):
        return {
            "params": int(self.params),
            "macs": int(self.macs),
            "memory_mb": float(self.memory_mb),
            "peak_activation_elements": int(self.peak_activation_elements),
        }


def _bn_params(c):
    return 4 * c


def estimate_architecture_cost(
    architecture,
    input_len=SEGMENT_LEN,
    num_classes=NUM_CLASSES,
):
    length = int(input_len)
    cin = 1
    params = 0
    macs = 0
    peak = length * cin

    stem = 8
    params += 7 * cin * stem + _bn_params(stem)
    macs += length * 7 * cin * stem
    peak = max(peak, length * stem)
    cin = stem

    for b in range(1, 5):
        kernels = architecture[f"block{b}_kernels"]
        cout = int(architecture[f"block{b}_channels"])
        branch = max(cout // 2, 4)

        for k in kernels:
            k = int(k)
            params += k * cin + cin * branch
            params += _bn_params(branch)
            macs += length * (k * cin + cin * branch)
            peak = max(peak, length * branch)

        params += cin * branch + _bn_params(branch)
        macs += length * cin * branch

        concat_c = 4 * branch
        peak = max(peak, length * concat_c)

        params += concat_c * cout + _bn_params(cout)
        macs += length * concat_c * cout
        peak = max(peak, length * cout)

        if cin != cout:
            params += cin * cout + _bn_params(cout)
            macs += length * cin * cout

        cin = cout

        if b < 4:
            length //= 2
            peak = max(peak, length * cin)

    head_in = 2 * cin

    params += head_in * 24 + _bn_params(24)
    macs += head_in * 24

    params += 24 * num_classes + num_classes
    macs += 24 * num_classes

    peak = max(peak, head_in, 24, num_classes)
    memory_mb = peak * 4 / (1024 ** 2)

    return ArchitectureCost(
        params=int(params),
        macs=int(macs),
        memory_mb=float(memory_mb),
        peak_activation_elements=int(peak),
    )
