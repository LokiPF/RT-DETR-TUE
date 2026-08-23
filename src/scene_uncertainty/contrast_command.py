"""Lightweight command handler for the reporting-only within-image contrast consumer."""

from __future__ import annotations

from .contrast_analysis import (
    build_anchor_diagnostics,
    build_contrast_rows,
    summarize_contrast_candidates,
)
from .contrast_controls import (
    attach_controls,
    rank_contrast_candidates,
    reference_control_rows,
)
from .contrast_inputs import load_contrast_inputs
from .contrast_reporting import write_contrast_report


class ContrastCommandError(ValueError):
    """An operator-actionable failure from the reporting-only command."""


def command_analyze_within_image_contrast(args) -> None:
    """Describe one completed tuning score bundle without importing the inference stack."""
    try:
        inputs = load_contrast_inputs(args.source)
        image_count = inputs.provenance["image_count"]
        rows, fits = build_contrast_rows(inputs)
        diagnostics = build_anchor_diagnostics(inputs)
        candidates = summarize_contrast_candidates(
            rows, expected_image_count=image_count
        )
        controls = summarize_contrast_candidates(
            reference_control_rows(rows), expected_image_count=image_count
        )
        attach_controls(candidates, controls, rows)
        ranking = rank_contrast_candidates(candidates)
        write_contrast_report(
            args.output, inputs=inputs, rows=rows, fits=fits, diagnostics=diagnostics,
            candidates=candidates, controls=controls, ranking=ranking,
        )
    except ValueError as error:
        raise ContrastCommandError(
            f"Cannot analyze within-image contrast: {error}"
        ) from error
