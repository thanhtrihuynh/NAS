from pynq import Overlay


BIT_FILE = (
    "overlay/ecg_accelerator.bit"
)


print(
    "Loading overlay..."
)

overlay = Overlay(
    BIT_FILE
)


print()
print(
    "Overlay loaded."
)

print()
print(
    "IP dictionary:"
)


for name, info in (
    overlay.ip_dict.items()
):
    print(
        name
    )


print()
print(
    "Hierarchy:"
)


for name in (
    overlay.hierarchy_dict.keys()
):
    print(
        name
    )


print()
print(
    "TEST OVERLAY DONE"
)