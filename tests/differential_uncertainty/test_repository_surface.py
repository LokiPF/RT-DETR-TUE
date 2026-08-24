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


def test_repository_has_at_most_45_python_files():
    python_files = [
        path
        for path in ROOT.glob("**/*.py")
        if ".git" not in path.relative_to(ROOT).parts
        and ".worktrees" not in path.relative_to(ROOT).parts
    ]
    assert len(python_files) <= 45
