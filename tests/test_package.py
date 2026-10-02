import degradation_monitor


def test_the_package_is_importable_and_versioned():
    assert degradation_monitor.__version__ == "2.0.0"


import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISTRIBUTIONS = {"PIL": "pillow", "sklearn": "scikit-learn", "uq_detr": "uq-detr"}  # import name -> pip name


def test_the_requirements_name_exactly_the_packages_the_code_imports():
    imported = set()
    for path in (ROOT / "degradation_monitor").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imported.add(node.module.split(".")[0])
    third_party = {DISTRIBUTIONS.get(name, name) for name in imported
                   if name not in sys.stdlib_module_names and name != "degradation_monitor"}
    lines = [line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    assert {re.split(r"[<>=]", line)[0] for line in lines} == third_party
