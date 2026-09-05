"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

from torch import nn

from ...core import register
from .rtdetrv2_decoder_tue import RTDETRTransformerv2TUE
from .tue_heads import TUEBase

__all__ = [
    "TUERTDETR",
]


@register()
class TUERTDETR(nn.Module):
    __inject__ = ["backbone", "encoder", "decoder", "tue_head"]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: RTDETRTransformerv2TUE,
        tue_head: TUEBase,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.tue_head = tue_head

    def forward_detector(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        return self.decoder(x, targets)

    def forward(self, x, targets=None):
        x = self.forward_detector(x, targets)

        if self.tue_head is not None:
            x = self.tue_head(x)

        return x

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
