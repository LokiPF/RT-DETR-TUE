from pathlib import Path

import torch


def print_structure(value, indent=0):
    prefix = " " * indent

    if isinstance(value, dict):
        print(f"{prefix}dict({len(value)})")
        for key, item in value.items():
            print(f"{prefix}  {key!r}:")
            print_structure(item, indent + 4)

    elif isinstance(value, torch.Tensor):
        finite = torch.isfinite(value) if value.is_floating_point() else None

        description = (
            f"Tensor(shape={tuple(value.shape)}, "
            f"dtype={value.dtype}, device={value.device}"
        )

        if finite is not None:
            description += (
                f", finite={finite.sum().item()}/{value.numel()}, "
                f"nan={torch.isnan(value).sum().item()}"
            )

        print(description + ")")

    elif isinstance(value, (list, tuple)):
        print(f"{prefix}{type(value).__name__}({len(value)})")
        for index, item in enumerate(value):
            print(f"{prefix}  [{index}]:")
            print_structure(item, indent + 4)

    else:
        print(f"{prefix}{type(value).__name__}: {value!r}")


path = Path("output/rtdetrv2_r18vd_120e_coco_tue_frechet_tests/frechet_means.pth")

state = torch.load(
    path,
    map_location="cpu",
    weights_only=False,
)

breakpoint()

print_structure(state)
