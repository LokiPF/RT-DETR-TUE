from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RETAINED_SRC = {
    "src/__init__.py",
    "src/nn/__init__.py",
    "src/nn/backbone/__init__.py",
    "src/nn/backbone/common.py",
    "src/nn/backbone/presnet.py",
    "src/zoo/__init__.py",
    "src/zoo/rtdetr/__init__.py",
    "src/zoo/rtdetr/box_ops.py",
    "src/zoo/rtdetr/denoising.py",
    "src/zoo/rtdetr/hybrid_encoder.py",
    "src/zoo/rtdetr/rtdetr.py",
    "src/zoo/rtdetr/rtdetrv2_decoder.py",
    "src/zoo/rtdetr/utils.py",
}


RETAINED_PYTHON = {
    "differential_uncertainty/__init__.py",
    "differential_uncertainty/__main__.py",
    "differential_uncertainty/artifacts.py",
    "differential_uncertainty/bank.py",
    "differential_uncertainty/cli.py",
    "differential_uncertainty/config.py",
    "differential_uncertainty/corruptions/__init__.py",
    "differential_uncertainty/corruptions/base.py",
    "differential_uncertainty/corruptions/gaussian_blur.py",
    "differential_uncertainty/evaluation.py",
    "differential_uncertainty/extraction.py",
    "differential_uncertainty/manifests.py",
    "differential_uncertainty/persistence.py",
    "differential_uncertainty/reporting.py",
    "differential_uncertainty/scoring.py",
    "src/__init__.py",
    "src/nn/__init__.py",
    "src/nn/backbone/__init__.py",
    "src/nn/backbone/common.py",
    "src/nn/backbone/presnet.py",
    "src/zoo/__init__.py",
    "src/zoo/rtdetr/__init__.py",
    "src/zoo/rtdetr/box_ops.py",
    "src/zoo/rtdetr/denoising.py",
    "src/zoo/rtdetr/hybrid_encoder.py",
    "src/zoo/rtdetr/rtdetr.py",
    "src/zoo/rtdetr/rtdetrv2_decoder.py",
    "src/zoo/rtdetr/utils.py",
    "tests/differential_uncertainty/test_artifacts.py",
    "tests/differential_uncertainty/test_bank.py",
    "tests/differential_uncertainty/test_cli.py",
    "tests/differential_uncertainty/test_config.py",
    "tests/differential_uncertainty/test_corruptions.py",
    "tests/differential_uncertainty/test_detector_parity.py",
    "tests/differential_uncertainty/test_evaluation.py",
    "tests/differential_uncertainty/test_extraction.py",
    "tests/differential_uncertainty/test_legacy_parity.py",
    "tests/differential_uncertainty/test_manifests.py",
    "tests/differential_uncertainty/test_persistence.py",
    "tests/differential_uncertainty/test_pipeline.py",
    "tests/differential_uncertainty/test_reporting.py",
    "tests/differential_uncertainty/test_repository_surface.py",
    "tests/differential_uncertainty/test_scoring.py",
}


def test_only_the_fixed_detector_closure_remains_under_src():
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src").glob("**/*.py")
    }
    assert actual == RETAINED_SRC


def test_training_and_annotation_dependencies_are_absent():
    requirements = (ROOT / "requirements.txt").read_text().lower()
    removed_dependencies = (
        "py" + "cocotools",
        "py" + "ya" + "ml",
        "tensor" + "board",
        "super" + "visely",
    )
    for removed in removed_dependencies:
        assert removed not in requirements


def test_repository_python_surface_is_exact():
    actual = set()
    for path in ROOT.glob("**/*.py"):
        relative = path.relative_to(ROOT)
        if relative.parts[0] in {".git", ".worktrees"}:
            continue
        actual.add(relative.as_posix())

    assert len(RETAINED_PYTHON) == 43
    assert actual == RETAINED_PYTHON
