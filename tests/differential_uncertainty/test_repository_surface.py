import importlib.machinery
import importlib.util
import os
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


RETAINED_IMPORT_DIRECTORIES = {
    "differential_uncertainty",
    "differential_uncertainty/corruptions",
    "src",
    "src/nn",
    "src/nn/backbone",
    "src/zoo",
    "src/zoo/rtdetr",
}
SOURCE_SUFFIXES = tuple(importlib.machinery.SOURCE_SUFFIXES)
BYTECODE_SUFFIXES = tuple(sorted(set(
    importlib.machinery.BYTECODE_SUFFIXES
    + importlib.machinery.DEBUG_BYTECODE_SUFFIXES
    + importlib.machinery.OPTIMIZED_BYTECODE_SUFFIXES
)))
EXTENSION_SUFFIXES = tuple(importlib.machinery.EXTENSION_SUFFIXES)
LEGACY_BYTECODE_SUFFIXES = (".pyo",)


def _repository_entries():
    pending = [ROOT]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(ROOT)
                if (
                    relative.parent == Path(".")
                    and relative.name in {".git", ".worktrees"}
                ):
                    continue
                symlink = entry.is_symlink()
                is_directory = entry.is_dir(follow_symlinks=False)
                yield path, relative, symlink, is_directory
                if is_directory and not symlink:
                    pending.append(path)


def _retained_cache_source(path):
    if path.parent.name != "__pycache__":
        return None
    candidates = [path]
    if "-pytest-" in path.name:
        standard_name = path.name.partition("-pytest-")[0] + ".pyc"
        candidates.append(path.with_name(standard_name))
    for candidate in candidates:
        try:
            source = Path(
                importlib.util.source_from_cache(str(candidate))
            )
            relative = source.relative_to(ROOT).as_posix()
        except (ValueError, TypeError):
            continue
        if relative in RETAINED_PYTHON and source.is_file():
            return relative
    return None


def _unexpected_import_artifacts():
    extras = set()
    for path, relative, symlink, is_directory in _repository_entries():
        if symlink or is_directory:
            continue
        name = path.name
        relative_name = relative.as_posix()
        if name.endswith(BYTECODE_SUFFIXES):
            if _retained_cache_source(path) is None:
                extras.add(relative_name)
        elif name.endswith(LEGACY_BYTECODE_SUFFIXES):
            extras.add(relative_name)
        elif name.endswith(EXTENSION_SUFFIXES):
            extras.add(relative_name)
    return extras


def test_only_the_fixed_detector_closure_remains_under_src():
    actual = {
        relative.as_posix()
        for path, relative, symlink, is_directory
        in _repository_entries()
        if (
            not symlink
            and not is_directory
            and relative.parts[0] == "src"
            and path.name.endswith(SOURCE_SUFFIXES)
        )
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
    actual = {
        relative.as_posix()
        for path, relative, symlink, is_directory
        in _repository_entries()
        if (
            not symlink
            and not is_directory
            and path.name.endswith(SOURCE_SUFFIXES)
        )
    }

    assert len(RETAINED_PYTHON) == 43
    assert actual == RETAINED_PYTHON


def test_only_retained_importable_artifacts_exist():
    assert _unexpected_import_artifacts() == set()


def test_importable_package_directory_topology_is_exact():
    actual = {
        relative.as_posix()
        for _path, relative, symlink, is_directory
        in _repository_entries()
        if (
            is_directory
            and not symlink
            and relative.parts[0]
            in {"src", "differential_uncertainty"}
            and "__pycache__" not in relative.parts
        )
    }

    assert actual == RETAINED_IMPORT_DIRECTORIES


def test_repository_tree_contains_no_symlinks():
    symlinks = set()
    pending = [ROOT]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(ROOT)
                if (
                    relative.parent == Path(".")
                    and relative.name in {".git", ".worktrees"}
                ):
                    continue
                if entry.is_symlink():
                    symlinks.add(relative.as_posix())
                elif entry.is_dir(follow_symlinks=False):
                    pending.append(path)

    assert symlinks == set()
