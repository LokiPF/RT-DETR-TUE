#!/usr/bin/env python3
"""Evaluate inverse-confidence AUROC for default RT-DETRv2."""

from evaluate_rtdetrv2_auroc_common import build_parser, run


def main() -> None:
    parser = build_parser(
        score_mode="default",
        description=(
            "Evaluate default RT-DETRv2 inverse-confidence ranking of "
            "matched localization failures."
        ),
    )
    run(parser.parse_args(), score_mode="default")


if __name__ == "__main__":
    main()

