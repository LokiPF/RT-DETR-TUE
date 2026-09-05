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


from dataclasses import dataclass, field

import torch


@dataclass
class FrechetAccumulator:
    sums: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    counts: dict[int, int] = field(default_factory=dict)

    def update(
        self,
        class_idx: int,
        layer_name: str,
        diagram: torch.Tensor,
    ):
        diagram = diagram.detach()

        if class_idx not in self.sums:
            self.sums[class_idx] = {}
            self.counts[class_idx] = 0

        if layer_name not in self.sums[class_idx]:
            self.sums[class_idx][layer_name] = torch.zeros_like(diagram)

        self.sums[class_idx][layer_name] += diagram

    def update_batch(
        self,
        class_indices: torch.Tensor,
        layer_name: str,
        diagrams: torch.Tensor,
    ):
        for c in class_indices.unique():
            mask = class_indices == c

            total = diagrams[mask].sum(dim=0)
            count = mask.sum()

            c = int(c.item())

            if c not in self.sums:
                self.sums[c] = {}

            if layer_name not in self.sums[c]:
                self.sums[c][layer_name] = torch.zeros_like(total)

            self.sums[c][layer_name] += total
            self.counts[c] = self.counts.get(c, 0) + int(count.item())

    def increment_class_count(self, class_idx: int, n: int = 1):
        self.counts[class_idx] = self.counts.get(class_idx, 0) + n

    def means(self):
        return {
            class_idx: {
                layer_name: total / self.counts[class_idx]
                for layer_name, total in layers.items()
            }
            for class_idx, layers in self.sums.items()
        }
