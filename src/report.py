"""Summary table of a folder of result files.

    python src/report.py results/

Runs that differ only by seed are grouped by cohort and configuration and shown
as a mean and a sample standard deviation over the seeds.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

from train import RESULT_PREFIX, RunResult, read_result


@dataclass
class MetricSummary:
    """One metric summarised over the seeds of one configuration."""

    mean: float
    deviation: float
    count: int

    def format(self, digits: int = 4) -> str:
        """Render as ``mean ± deviation``, or just the mean for a single run."""
        if self.count < 2:
            return f"{self.mean:.{digits}f}"
        return f"{self.mean:.{digits}f} ± {self.deviation:.{digits}f}"


@dataclass
class RunGroup:
    """Every run of one cohort and variant."""

    cohort: str
    variant: str
    runs: list[RunResult]

    @property
    def seeds(self) -> list[int]:
        """Seeds present in this group, in order."""
        return sorted(run.seed for run in self.runs)

    def summarise(self, metric: str) -> MetricSummary | None:
        """Summarise one metric over the seeds, or return None if it is absent."""
        values = [run.metrics[metric] for run in self.runs if metric in run.metrics]
        if not values:
            return None
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        return MetricSummary(mean=statistics.fmean(values), deviation=deviation, count=len(values))

    def baseline(self, metric: str) -> float | None:
        """The copy-forward baseline's value for one metric, which every seed shares."""
        values = [run.baseline[metric] for run in self.runs if metric in run.baseline]
        return statistics.fmean(values) if values else None


def find_results(root: Path) -> list[Path]:
    """List every result file under ``root``, searching subdirectories."""
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(root.rglob(f"{RESULT_PREFIX}*.json"))


def load_runs(root: Path) -> list[RunResult]:
    """Read every result file under ``root``.

    Raises:
        FileNotFoundError: no result file was found.
    """
    paths = find_results(root)
    if not paths:
        raise FileNotFoundError(f"No result files found under {root}.")
    return [read_result(path) for path in paths]


def group_runs(runs: list[RunResult]) -> list[RunGroup]:
    """Group runs by cohort and variant, keeping the cohort order of first sight."""
    order: list[tuple[str, str]] = []
    grouped: dict[tuple[str, str], list[RunResult]] = {}
    for run in runs:
        key = (run.cohort, run.variant)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(run)
    return [RunGroup(cohort=cohort, variant=variant, runs=grouped[(cohort, variant)])
            for cohort, variant in order]


METRIC_LABELS = {
    "jaccard": "Jaccard",
    "f1": "F1",
    "average_precision": "PRAUC",
    "interaction_rate": "DDI rate",
    "change_started": "NHJ_new",
    "change_stopped": "NHJ_dropped",
    "precision": "Precision",
    "recall": "Recall",
    "drugs_recommended": "Drugs out",
    "drugs_recorded": "Drugs recorded",
}
DEFAULT_METRICS = (
    "jaccard",
    "f1",
    "average_precision",
    "interaction_rate",
    "change_started",
    "change_stopped",
)
LOWER_IS_BETTER = ("interaction_rate",)


def _header(metrics: tuple[str, ...], with_baseline: bool) -> list[str]:
    """Column names of the table."""
    columns = ["Cohort", "Variant", "Seeds"]
    columns += [METRIC_LABELS.get(metric, metric) for metric in metrics]
    if with_baseline:
        columns.append("Copy-forward Jaccard")
    return columns


def _row(group: RunGroup, metrics: tuple[str, ...], with_baseline: bool) -> list[str]:
    """One table row for one configuration."""
    cells = [group.cohort, group.variant, str(len(group.runs))]
    for metric in metrics:
        summary = group.summarise(metric)
        cells.append(summary.format() if summary else "not reported")
    if with_baseline:
        baseline = group.baseline("jaccard")
        cells.append("not reported" if baseline is None else f"{baseline:.4f}")
    return cells


def build_rows(
    groups: list[RunGroup],
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    with_baseline: bool = True,
) -> tuple[list[str], list[list[str]]]:
    """Return the header and the rows of the comparison table."""
    return (
        _header(metrics, with_baseline),
        [_row(group, metrics, with_baseline) for group in groups],
    )


def as_text(header: list[str], rows: list[list[str]]) -> str:
    """Render the table with aligned columns for a terminal."""
    widths = [
        max(len(header[column]), *(len(row[column]) for row in rows))
        if rows
        else len(header[column])
        for column in range(len(header))
    ]
    lines = [
        "  ".join(name.ljust(width) for name, width in zip(header, widths, strict=True)),
        "  ".join("-" * width for width in widths),
    ]
    lines += [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in rows
    ]
    note = (
        "Values are the mean over seeds plus or minus the sample standard deviation. "
        "Lower is better for the DDI rate; higher is better for every other column."
    )
    return "\n".join([*lines, "", note])


def as_csv(header: list[str], rows: list[list[str]]) -> str:
    """Render the table as comma-separated values."""
    def escape(cell: str) -> str:
        return f'"{cell}"' if "," in cell else cell

    return "\n".join(
        ",".join(escape(cell) for cell in line) for line in [header, *rows]
    )


def as_markup(header: list[str], rows: list[list[str]]) -> str:
    """Render the table in the pipe-delimited form a manuscript can take."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


FORMATTERS = {"text": as_text, "csv": as_csv, "markup": as_markup}


def render(
    groups: list[RunGroup],
    output_format: str = "text",
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    with_baseline: bool = True,
) -> str:
    """Render grouped results in one of the three formats.

    Raises:
        ValueError: the format is not one of ``text``, ``csv`` or ``markup``.
    """
    if output_format not in FORMATTERS:
        raise ValueError(f"Unknown format '{output_format}'. Choose from {sorted(FORMATTERS)}.")
    header, rows = build_rows(groups, metrics=metrics, with_baseline=with_baseline)
    return FORMATTERS[output_format](header, rows)


def main(argv: list[str] | None = None) -> int:
    """Print one row per configuration: the mean and spread over its seeds."""
    parser = argparse.ArgumentParser(description="Summarise a folder of result files.")
    parser.add_argument("results", type=Path, nargs="?", default=Path("results"),
                        help="folder holding result files, searched recursively")
    parser.add_argument("--format", dest="output_format", default="text",
                        choices=("text", "csv", "markup"))
    parser.add_argument("--output", type=Path, default=None,
                        help="write to this file instead of the terminal")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS),
                        choices=sorted(METRIC_LABELS), metavar="METRIC")
    parser.add_argument("--cohort", default=None, help="show one cohort only")
    parser.add_argument("--no_baseline", action="store_true",
                        help="leave out the copy-forward baseline column")
    arguments = parser.parse_args(argv)

    try:
        runs = load_runs(arguments.results)
    except FileNotFoundError as error:
        print(error)
        return 1
    if arguments.cohort:
        runs = [run for run in runs if run.cohort == arguments.cohort]
        if not runs:
            print(f"No runs found for cohort '{arguments.cohort}'.")
            return 1
    table = render(
        group_runs(runs),
        output_format=arguments.output_format,
        metrics=tuple(arguments.metrics),
        with_baseline=not arguments.no_baseline,
    )
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(table + "\n", encoding="utf-8")
        print(f"Written to {arguments.output}")
    else:
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
