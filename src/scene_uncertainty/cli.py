"""The one entry point for the class-independent persistence scene-uncertainty study.

Six subcommands, run in this order:

    select            -> reference.json + evaluation.json (which COCO images are used)
    extract-reference -> a feature cache over the clean reference images
    extract-blur      -> a feature cache over the evaluation images, all six blur levels
    build-bank        -> a class-independent query bank per decoder layer
    evaluate-knn      -> scored rows + per-query distances + a result manifest
    report            -> summary.json and the trend plots

Argument validation lives here rather than in `pipeline`, so a typo in a policy name,
an unreadable checkpoint, or a negative batch size fails at parse time instead of an
hour into a GPU run. `pipeline` is imported lazily so `--help` costs no torch import.
"""

from __future__ import annotations

import argparse
import sys


_SELECT_EPILOG = """\
Writes `<output>/reference.json` (bank images) and `<output>/evaluation.json` (scored
images, split into tuning and test). Both files record the seed they were drawn with;
the extract commands copy it into their artifact metadata.

The rare-class greedy pass in the reference selection scans every candidate image once
per pick, so it is the slow half of this command. It prints its own wall time.
"""

_EXTRACT_EPILOG = """\
ONE WRITER PER OUTPUT DIRECTORY. The artifact's immutability guard is a file check, so
two extractions pointed at the same `--output` at the same time both pass it: the second
silently resumes the first's partial manifest and both write the same shard index, which
loses records with no error. Give every concurrent extraction its own `--output`.

Resume is supported: rerun the identical command and the shards already flushed are kept.
Recovering which records exist deserializes every completed shard, tensor payloads and
all, so resuming a multi-gigabyte extraction pays one full read of what it already wrote
before it extracts anything new. The scan reports its own wall time on stderr.

Resume requires the same git commit, checkpoint, config, annotation file and image list
as the interrupted run: the features in one artifact must come from one version of the
extraction code. A mismatch is reported by name and nothing is written.
"""

_BUILD_BANK_EPILOG = """\
`--population natural` samples only the uniformly drawn reference images; `coverage` also
uses the rare-class augmentation images and balances matched-object against background
queries. The bank is class-independent either way: no class label survives into it.
"""

_EVALUATE_EPILOG = """\
`--output` is the results CSV path. Three siblings are written next to it:
`<output>.query_distances.pt`, `<output>.normalizers.pt`, and `<output>.manifest.json`.
The manifest carries the clean-distance fit for every layer -- `center`, `scale`,
`min_neighbor_distance` and `near_duplicate_fraction` -- so the fit can be judged without
unpickling anything. A `near_duplicate_fraction` above ~0.01 means near-duplicate bank
rows pulled the fit, and every score in the CSV is divided by that `scale`.

`--bank-chunk-size` only trades memory against speed. It is recorded for provenance and
deliberately kept out of the artifact id: two runs that differ only in it are the same run.
"""

_REPORT_EPILOG = """\
`--results` is the CSV written by evaluate-knn; its `<results>.manifest.json` sibling is
read for the run metadata and must sit next to it.
"""


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must not be negative, got {value}")
    return value


def _name_list(kind: str):
    """Validate a comma-separated policy or aggregation list at parse time.

    The known names live in `pipeline`, which drags in torch, matplotlib and the detector
    zoo; importing them here would make `--help` pay for all of it. The import happens
    when a list is actually parsed, which is only ever immediately before `pipeline` is
    imported anyway.
    """

    def convert(text: str):
        from .pipeline import AGGREGATIONS, POLICIES

        allowed = POLICIES if kind == "policy" else AGGREGATIONS
        names = tuple(value.strip() for value in text.split(",") if value.strip())
        if not names:
            raise argparse.ArgumentTypeError(f"needs at least one {kind} name")
        unknown = sorted(set(names) - set(allowed))
        if unknown:
            raise argparse.ArgumentTypeError(
                f"unknown {kind} names {unknown}; choose from {list(allowed)}"
            )
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise argparse.ArgumentTypeError(f"repeated {kind} names {duplicated}")
        return names

    return convert


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scene_uncertainty",
        description="Class-independent persistence scene uncertainty",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str, epilog: str) -> argparse.ArgumentParser:
        return subparsers.add_parser(
            name,
            help=help_text,
            description=help_text,
            epilog=epilog,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

    select = add("select", "Draw the reference and evaluation image ids.", _SELECT_EPILOG)
    select.add_argument("--train-ann", required=True, help="COCO train2017 instances JSON")
    select.add_argument("--val-ann", required=True, help="COCO val2017 instances JSON")
    select.add_argument("--output", required=True, help="directory for the two selection files")
    select.add_argument("--seed", type=_non_negative_int, default=42,
                        help="numpy seed; the selection files record it for the extractors")
    select.add_argument("--natural-count", type=_positive_int, default=4000,
                        help="uniformly drawn reference images")
    select.add_argument("--augmentation-budget", type=_positive_int, default=1000,
                        help="extra reference images drawn to fill rare-class quotas")
    select.add_argument("--quota", type=_positive_int, default=50,
                        help="reference images each COCO category should reach")

    for name, reference in (("extract-reference", True), ("extract-blur", False)):
        extract = add(
            name,
            "Extract classification persistence for the {} images.".format(
                "clean reference" if reference else "blurred evaluation"
            ),
            _EXTRACT_EPILOG,
        )
        extract.add_argument("--config", required=True, help="model-only YAML config")
        extract.add_argument("--checkpoint", required=True, help="detector checkpoint")
        extract.add_argument("--images", required=True, help="COCO image directory")
        extract.add_argument("--annotations", required=True, help="COCO instances JSON")
        extract.add_argument(
            "--selection",
            required=True,
            help="reference.json" if reference else "evaluation.json",
        )
        extract.add_argument("--output", required=True, help="feature cache directory (one writer only)")
        extract.add_argument("--device", default="cuda:0")
        extract.add_argument("--limit", type=_positive_int, help="extract only the first N images")
        extract.add_argument("--batch-size", type=_positive_int, default=1)
        extract.add_argument("--num-workers", type=_non_negative_int, default=4)
        extract.add_argument("--shard-size", type=_positive_int, default=50,
                             help="records per shard; also the resume granularity")

    build_bank = add("build-bank", "Build the class-independent query bank.", _BUILD_BANK_EPILOG)
    build_bank.add_argument("--cache", required=True, help="reference feature cache directory")
    build_bank.add_argument("--output", required=True, help="bank directory")
    build_bank.add_argument("--population", choices=("natural", "coverage"), required=True)
    build_bank.add_argument("--capacity", type=_positive_int, required=True,
                            help="vectors per decoder layer")
    build_bank.add_argument("--seed", type=_non_negative_int, default=42,
                            help="offset by the layer id, so each layer bank draws independently")

    evaluate = add("evaluate-knn", "Score cached scenes against the bank.", _EVALUATE_EPILOG)
    evaluate.add_argument("--cache", required=True, help="evaluation feature cache directory")
    evaluate.add_argument("--bank", required=True, help="bank directory")
    evaluate.add_argument("--output", required=True, help="results CSV path")
    evaluate.add_argument(
        "--normalization",
        choices=("raw", "robust_z", "unit", "shape_scale"),
        required=True,
    )
    evaluate.add_argument("--partition", choices=("tuning", "test", "all"), default="tuning")
    evaluate.add_argument(
        "--policies",
        type=_name_list("policy"),
        default="all,top10,top20,top50,threshold_0.2,threshold_0.3,threshold_0.5,smooth_1,smooth_2,"
                "oracle_matched,oracle_background,oracle_correct,oracle_incorrect",
    )
    evaluate.add_argument("--aggregations", type=_name_list("aggregation"),
                          default="mean,median,q90,top20_mean")
    evaluate.add_argument("--k", type=_positive_int, default=5)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--bank-chunk-size", type=_positive_int, default=8192,
                          help="kNN bank chunk width; performance only, not part of the artifact id")

    report = add("report", "Summarize scored rows into plots and summary.json.", _REPORT_EPILOG)
    report.add_argument("--results", required=True, help="results CSV written by evaluate-knn")
    report.add_argument("--output", required=True, help="report directory")
    return parser


def main(argv=None) -> int:
    """Run one subcommand. Returns the process exit status.

    A failure the operator caused -- a missing artifact, an incompatible cache, a partition
    that selected nothing -- is reported as one line on stderr, not as a traceback, because
    the traceback says nothing the message does not and buries it under twenty frames of
    torch. Everything else propagates: an unexpected exception is a bug and should look
    like one.
    """
    args = build_parser().parse_args(argv)
    from .pipeline import COMMANDS, PipelineError

    try:
        COMMANDS[args.command](args)
    except (PipelineError, FileExistsError, FileNotFoundError) as error:
        print(f"scene_uncertainty {args.command}: error: {error}", file=sys.stderr)
        return 2
    return 0
