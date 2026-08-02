"""Tests for the one figure whose bars do not share a provenance.

``censorship_vs_orthogonal.png`` ranks the full-space run's censorship results against a *separate*
``--orthogonalize-country`` run's. That cross-run comparison is the figure's purpose, but nothing
in the bars says so: two runs can be weeks and a re-embed apart, and the existing baseline check
only proves they scored the same validation split.

So the caption has to name which report each series came from, and a comparison over model sets
that differ between the runs has to be refused rather than drawn -- otherwise a model missing from
one side reads as a model that simply performed differently.
"""
import pathlib

import pytest

from working_paper_authorship import authorship_performance_figures as figs


def _report(path: pathlib.Path, models=("Logistic Regression", "Random Forest"),
            datasets=("raw__full", "naive__full", "llm_censorship__full"), baseline=0.5) -> pathlib.Path:
    lines = [f"Random-guess BCE baseline (class base rates): {baseline:.4f}", ""]
    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            lines.append(f"{model:22s} {dataset:24s} {0.30 + 0.01 * (i + j):.4f} {0.5:.4f} x")
    path.write_text("\n".join(lines))
    return path


def test_reads_back_the_models_and_datasets_it_was_given(tmp_path):
    rows, baseline = figs.read_report(_report(tmp_path / "report.txt"))
    assert baseline == 0.5
    assert {r["model"] for r in rows} == {"Logistic Regression", "Random Forest"}
    assert {r["dataset"] for r in rows} == {"raw__full", "naive__full", "llm_censorship__full"}


def test_comparison_figure_names_both_source_reports(tmp_path):
    """The fix for the provenance gap: a reader can see the two series came from different runs,
    and when each was written."""
    full = _report(tmp_path / "report.txt")
    orth = _report(tmp_path / "orthogonal.txt")

    captured = {}
    real_render = figs.render

    def _capture(bars, baseline, title, subtitle, path):
        captured[path.name] = subtitle
        return real_render(bars, baseline, title, subtitle, path)

    figs.render = _capture
    try:
        figs.render_all_figures(full, orth, tmp_path / "figures")
    finally:
        figs.render = real_render

    subtitle = captured["censorship_vs_orthogonal.png"]
    assert "report.txt" in subtitle and "orthogonal.txt" in subtitle
    assert "two separate runs" in subtitle
    assert figs._written_at(full) in subtitle


def test_differing_model_sets_are_refused(tmp_path):
    """A side-by-side ranking over different model sets is not a comparison."""
    full = _report(tmp_path / "report.txt", models=("Logistic Regression", "Random Forest"))
    orth = _report(tmp_path / "orthogonal.txt", models=("Logistic Regression",))

    with pytest.raises(ValueError, match="Model sets differ"):
        figs.render_all_figures(full, orth, tmp_path / "figures")


def test_differing_baselines_are_refused(tmp_path):
    """Pre-existing guard, kept: both runs score the same validation split, so a disagreeing
    no-skill reference means the reports describe different data."""
    full = _report(tmp_path / "report.txt", baseline=0.5)
    orth = _report(tmp_path / "orthogonal.txt", baseline=0.9)

    with pytest.raises(ValueError, match="Baseline mismatch"):
        figs.render_all_figures(full, orth, tmp_path / "figures")


def test_the_single_run_figures_render_without_an_orthogonal_report(tmp_path):
    written = figs.render_all_figures(_report(tmp_path / "report.txt"),
                                      tmp_path / "absent.txt", tmp_path / "figures")
    names = {p.name for p in written}
    assert names == {"raw_methods.png", "censorship_methods.png"}
    assert all(p.exists() for p in written)


def test_the_comparison_figure_is_written_when_both_runs_are_present(tmp_path):
    written = figs.render_all_figures(_report(tmp_path / "report.txt"),
                                      _report(tmp_path / "orthogonal.txt"),
                                      tmp_path / "figures")
    assert (tmp_path / "figures" / "censorship_vs_orthogonal.png").exists()
    assert len(written) == 3
