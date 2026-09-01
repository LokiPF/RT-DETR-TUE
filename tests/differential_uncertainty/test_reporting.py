import csv
import copy
import json

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pytest

from differential_uncertainty.corruptions import CORRUPTION_NAMES
from differential_uncertainty.evaluation import evaluate_scores
from differential_uncertainty.reporting import write_results
from tests.differential_uncertainty.test_evaluation import FAMILIES, _rows


def _inputs():
    rows = _rows()
    evaluation = evaluate_scores(rows, FAMILIES, samples=30, seed=4)
    return rows, evaluation


def test_write_results_writes_exactly_the_five_required_files_and_csv_schemas(tmp_path):
    rows, evaluation = _inputs()
    output = tmp_path / "evidence"
    write_results(output, rows, evaluation, FAMILIES)

    assert sorted(path.name for path in output.iterdir()) == [
        "corruption_auroc_bars.png", "per_image_scores.csv", "report.md",
        "results.csv", "summary.json",
    ]
    with (output / "per_image_scores.csv").open(newline="") as handle:
        score_rows = list(csv.reader(handle))
    assert score_rows[0] == [
        "image_id", "corruption", "severity", "fingerprint", "confidence", "entropy"
    ]
    assert len(score_rows) == 1 + len(rows)
    assert [(row[1], row[0], int(row[2])) for row in score_rows[1:]] == [
        (family, image_id, severity)
        for family in FAMILIES
        for image_id in ("a", "b", "c")
        for severity in (0, 4, 5)
    ]
    with (output / "results.csv").open(newline="") as handle:
        task_rows = list(csv.reader(handle))
    assert task_rows[0] == [
        "corruption", "severity", "fingerprint_auroc", "confidence_auroc", "entropy_auroc"
    ]
    assert len(task_rows) == 39
    assert json.loads((output / "summary.json").read_text()) == evaluation


def test_report_has_all_family_rows_real_chart_and_required_final_line(tmp_path):
    rows, evaluation = _inputs()
    output = tmp_path / "evidence"
    write_results(output, rows, evaluation, FAMILIES)

    text = (output / "report.md").read_text()
    assert "mean cosine distance to the five nearest bank vectors" in text
    assert "confidence-weighted mean across the retained scene queries" in text
    assert "one minus the maximum sigmoid confidence across retained queries" in text
    assert "normalized Shannon entropy of the highest-confidence retained query" in text
    assert "model confidence score" not in text.lower()
    assert "2.5th and 97.5th percentiles from paired whole-image bootstrap resampling of the supplied evaluation set" in text
    assert "descriptive stability intervals" in text
    assert "equal-weight macro-AUROC difference" in text
    assert "Levels 1 through 3 were not evaluated." in text
    assert all(f"| {family} |" in text for family in FAMILIES)
    assert text.rstrip().splitlines()[-1] == "![Per-corruption AUROC](corruption_auroc_bars.png)"
    assert not any(word in text.lower() for word in (
        "spearman", "orientation", "conditional", "candidate", "trend", "levels 1-3"
    ))
    chart = output / "corruption_auroc_bars.png"
    assert chart.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    pixels = mpimg.imread(chart)
    assert pixels.shape[0] > 100 and pixels.shape[1] > 100


def test_chart_closes_its_figure_when_saving_fails(tmp_path, monkeypatch):
    rows, evaluation = _inputs()
    before = set(plt.get_fignums())

    def fail_savefig(_self, *_args, **_kwargs):
        raise RuntimeError("save failed")

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", fail_savefig)
    with pytest.raises(RuntimeError, match="save failed"):
        write_results(tmp_path / "evidence", rows, evaluation, FAMILIES)
    assert set(plt.get_fignums()) == before


@pytest.mark.parametrize("families", [
    CORRUPTION_NAMES[:1],
    (CORRUPTION_NAMES[1], CORRUPTION_NAMES[0], *CORRUPTION_NAMES[2:]),
])
def test_write_results_requires_the_exact_fixed_family_roster_in_order(tmp_path, families):
    rows, evaluation = _inputs()
    with pytest.raises(ValueError, match="fixed corruption roster"):
        write_results(tmp_path / "evidence", rows, evaluation, families)


def test_write_results_accepts_and_normalizes_integer_valued_real_scores(tmp_path):
    rows, evaluation = _inputs()
    for row in rows:
        for method in ("fingerprint", "confidence", "entropy"):
            row[method] = int(row[method])
    output = tmp_path / "evidence"
    write_results(output, rows, evaluation, FAMILIES)

    with (output / "per_image_scores.csv").open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        first_score = next(reader)
    assert first_score[3:] == ["0.0", "0.0", "0.0"]


def test_report_interprets_paired_intervals_honestly(tmp_path):
    rows, evaluation = _inputs()
    evaluation = copy.deepcopy(evaluation)
    evaluation["comparisons"] = {
        "fingerprint_minus_confidence": {"point": 0.1, "low": 0.01, "high": 0.2},
        "fingerprint_minus_entropy": {"point": -0.1, "low": -0.2, "high": -0.01},
    }
    output = tmp_path / "evidence"
    write_results(output, rows, evaluation, FAMILIES)

    text = (output / "report.md").read_text().lower()
    assert "fingerprint advantage over confidence" in text
    assert "entropy advantage over fingerprint" in text

    evaluation["comparisons"]["fingerprint_minus_confidence"] = {
        "point": 0.0, "low": -0.01, "high": 0.01
    }
    write_results(output, rows, evaluation, FAMILIES)
    text = (output / "report.md").read_text().lower()
    assert "fingerprint versus confidence: inconclusive" in text
    assert "fingerprint advantage over confidence" not in text
