from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from torch import nn


def tensor_summary(tensor: torch.Tensor) -> str:
    if tensor.is_floating_point() or tensor.is_complex():
        finite = torch.isfinite(tensor).sum().item()
    else:
        finite = tensor.numel()

    return (
        f"Tensor(shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, "
        f"device={tensor.device}, "
        f"finite={finite})"
    )


@dataclass(slots=True, eq=False)
class LayerCapture:
    input: torch.Tensor
    weight: torch.Tensor
    output: torch.Tensor

    def __repr__(self) -> str:
        return (
            "LayerCapture(\n"
            f"  input:  {tensor_summary(self.input)}\n"
            f"  weight: {tensor_summary(self.weight)}\n"
            f"  output: {tensor_summary(self.output)}\n"
            ")"
        )


@dataclass(slots=True, eq=False, repr=False)
class CaptureGroup:
    selected: dict[nn.Module, str]
    data: dict[str, LayerCapture] = field(default_factory=dict)

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        selected_modules: Iterable[nn.Module],
    ) -> "CaptureGroup":
        all_names = {module: name for name, module in model.named_modules()}

        selected_modules = tuple(selected_modules)

        missing = [module for module in selected_modules if module not in all_names]
        if missing:
            raise ValueError(f"{len(missing)} modules are not part of the model")

        return cls({module: all_names[module] for module in selected_modules})

    def clear(self) -> None:
        self.data.clear()

    def add(
        self,
        module: nn.Module,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
    ) -> None:
        try:
            name = self.selected[module]
        except KeyError:
            return

        self.data[name] = LayerCapture(
            input=input_tensor.detach(),
            weight=module.weight.detach(),
            output=output_tensor.detach(),
        )

    def __repr__(self) -> str:
        if not self.data:
            return "CaptureGroup()"

        lines = ["CaptureGroup("]

        for layer_name, capture in self.data.items():
            lines.extend(
                [
                    f"  {layer_name}:",
                    f"    input:  {tensor_summary(capture.input)}",
                    f"    weight: {tensor_summary(capture.weight)}",
                    f"    output: {tensor_summary(capture.output)}",
                ]
            )

        lines.append(")")
        return "\n".join(lines)


@dataclass
class DiagramStatistics:
    sums: torch.Tensor
    counts: torch.Tensor
