"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

from ....core import register
from .tue_base import TUEBase

__all__ = ["TUEOrig"]


@register()
class TUEOrig(TUEBase):
    def __init__(
        self,
        frechet_means: str,
        num_top_queries: int,
        num_classes: int,
        confidence_threshold: float = 0.0,
    ):
        super().__init__(frechet_means, num_top_queries, num_classes)

        self.confidence_threshold = confidence_threshold

    def forward(
        self,
        outputs: dict,
    ) -> dict:
        results = self.calculate_tu(outputs, self.confidence_threshold)
        return results
