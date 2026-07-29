#!/usr/bin/env python3
"""Compare learned bbox uncertainty against default RT-DETRv2 confidence."""

from evaluate_rtdetrv2_auroc_common import build_parser, run


def main() -> None:
    parser = build_parser(
        score_mode="learned",
        description=(
            "Evaluate learned total bbox uncertainty and compare it with "
            "default RT-DETRv2 inverse confidence on the same detections."
        ),
    )
    run(parser.parse_args(), score_mode="learned")


if __name__ == "__main__":
    main()
