"""Write data-cleaning and validation summaries."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TABLE_NAMES = ("track", "nodes", "arcs", "route")


@dataclass
class TripDataSummary:
    trip_id: str
    raw_table_rows: dict[str, int]
    cleaned_table_rows: dict[str, int] | None
    invalid_track_rows: int
    duplicate_track_rows: int
    raw_validation_issues: tuple[str, ...]
    post_clean_validation_issues: tuple[str, ...]
    status: str


@dataclass
class PartitionDataSummary:
    name: str
    source_dir: Path
    trips: list[TripDataSummary]
    candidate_rows: int
    candidate_coverage: float
    observation_count: int
    sampled_observation_count: int
    model_trip_count: int


def _format_count(value: int) -> str:
    return f"{value:,}"


def _partition_totals(
    partition: PartitionDataSummary,
    table_name: str,
    cleaned: bool,
) -> int:
    total = 0
    for trip in partition.trips:
        rows = trip.cleaned_table_rows if cleaned else trip.raw_table_rows
        if rows is not None:
            total += rows[table_name]
    return total


def _write_summary_chart(
    partitions: list[PartitionDataSummary],
    output_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    names = [partition.name for partition in partitions]
    positions = list(range(len(partitions)))
    raw_track_rows = [
        _partition_totals(partition, "track", cleaned=False)
        for partition in partitions
    ]
    cleaned_track_rows = [
        _partition_totals(partition, "track", cleaned=True)
        for partition in partitions
    ]
    invalid_rows = [
        sum(trip.invalid_track_rows for trip in partition.trips)
        for partition in partitions
    ]
    duplicate_rows = [
        sum(trip.duplicate_track_rows for trip in partition.trips)
        for partition in partitions
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    width = 0.36
    axes[0].bar(
        [position - width / 2 for position in positions],
        raw_track_rows,
        width,
        label="Loaded",
        color="#4C78A8",
    )
    axes[0].bar(
        [position + width / 2 for position in positions],
        cleaned_track_rows,
        width,
        label="After cleaning",
        color="#59A14F",
    )
    axes[0].set_title("GPS observations before and after cleaning")
    axes[0].set_ylabel("Track rows")
    axes[0].set_xticks(positions, names)
    axes[0].yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value:,.0f}")
    )
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        [position - width / 2 for position in positions],
        invalid_rows,
        width,
        label="Invalid rows",
        color="#E15759",
    )
    axes[1].bar(
        [position + width / 2 for position in positions],
        duplicate_rows,
        width,
        label="Duplicate rows",
        color="#F28E2B",
    )
    axes[1].set_title("GPS observations removed")
    axes[1].set_ylabel("Rows removed")
    axes[1].set_xticks(positions, names)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    if not any(invalid_rows) and not any(duplicate_rows):
        axes[1].set_ylim(0, 1)
        for position in positions:
            axes[1].text(
                position,
                0.03,
                "0 removed",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    figure.tight_layout()
    chart_path = output_dir / "data_quality_summary.png"
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)
    return chart_path


def _markdown_report(
    partitions: list[PartitionDataSummary],
    chart_path: Path,
    sampling_description: str,
) -> str:
    lines = [
        "# Data cleaning and validation report",
        "",
        "This report summarizes the supplied partitions after loading, cleaning, "
        "and validation. Raw validation issues are reported before cleaning; "
        "post-clean issues are checked before feature generation.",
        "",
        "## Partition summary",
        "",
        "| Partition | Trip folders | Cleaned/validated | Used for modeling | Dropped: empty tables | Raw track rows | Clean track rows | Track retained | Invalid removed | Duplicates removed | Post-clean issues | Candidate rows | Candidate coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for partition in partitions:
        dropped_empty = sum(
            trip.status.startswith("Dropped: empty required table(s)")
            for trip in partition.trips
        )
        cleaned_and_validated = sum(
            trip.cleaned_table_rows is not None
            and not trip.post_clean_validation_issues
            for trip in partition.trips
        )
        invalid_rows = sum(
            trip.invalid_track_rows for trip in partition.trips
        )
        duplicate_rows = sum(
            trip.duplicate_track_rows for trip in partition.trips
        )
        post_clean_issues = sum(
            len(trip.post_clean_validation_issues)
            for trip in partition.trips
        )
        raw_track_rows = _partition_totals(partition, "track", cleaned=False)
        clean_track_rows = _partition_totals(partition, "track", cleaned=True)
        track_retention = (
            clean_track_rows / raw_track_rows if raw_track_rows else 0.0
        )
        lines.append(
            f"| {partition.name} | {_format_count(len(partition.trips))} | "
            f"{_format_count(cleaned_and_validated)} | "
            f"{_format_count(partition.model_trip_count)} | "
            f"{_format_count(dropped_empty)} | "
            f"{_format_count(raw_track_rows)} | "
            f"{_format_count(clean_track_rows)} | "
            f"{track_retention:.2%} | "
            f"{_format_count(invalid_rows)} | "
            f"{_format_count(duplicate_rows)} | "
            f"{_format_count(post_clean_issues)} | "
            f"{_format_count(partition.candidate_rows)} | "
            f"{partition.candidate_coverage:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Track cleaning figures",
            "",
            f"![Loaded and cleaned observations, including removed-row counts]"
            f"({chart_path.name})",
            "",
            "## Table row counts",
            "",
            "Loaded counts include all discovered trips. After-cleaning counts "
            "include trips that passed required-table checks and were cleaned; "
            "dropped trips are excluded.",
            "",
            "| Partition | Table | Loaded rows | After cleaning |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for partition in partitions:
        for table_name in TABLE_NAMES:
            lines.append(
                f"| {partition.name} | {table_name} | "
                f"{_format_count(_partition_totals(partition, table_name, False))} | "
                f"{_format_count(_partition_totals(partition, table_name, True))} |"
            )

    raw_issue_counts: dict[str, Counter[str]] = {}
    post_clean_issue_counts: dict[str, Counter[str]] = {}
    for partition in partitions:
        raw_issue_counts[partition.name] = Counter(
            issue
            for trip in partition.trips
            for issue in trip.raw_validation_issues
        )
        post_clean_issue_counts[partition.name] = Counter(
            issue
            for trip in partition.trips
            for issue in trip.post_clean_validation_issues
        )

    lines.extend(
        [
            "",
            "## Validation findings before cleaning",
            "",
            "| Partition | Issue found | Trips reporting issue |",
            "| --- | --- | ---: |",
        ]
    )
    has_raw_issues = False
    for partition in partitions:
        for issue, count in sorted(raw_issue_counts[partition.name].items()):
            has_raw_issues = True
            escaped_issue = issue.replace("|", "\\|")
            lines.append(
                f"| {partition.name} | {escaped_issue} | "
                f"{_format_count(count)} |"
            )
    if not has_raw_issues:
        lines.append("| All partitions | None | 0 |")

    lines.extend(
        [
            "",
            "## Validation findings after cleaning",
            "",
        ]
    )
    has_post_clean_issues = False
    for partition in partitions:
        for issue, count in sorted(
            post_clean_issue_counts[partition.name].items()
        ):
            has_post_clean_issues = True
            lines.append(
                f"- **{partition.name}:** {issue} "
                f"({count} occurrence(s))"
            )
    if not has_post_clean_issues:
        lines.append(
            "No remaining validation issues were found in trips that passed "
            "cleaning."
        )

    lines.extend(
        [
            "",
            "## Per-trip audit",
            "",
            "| Partition | Trip | Status | Track rows (loaded → cleaned) | Invalid removed | Duplicates removed | Raw validation issues |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for partition in partitions:
        for trip in partition.trips:
            cleaned_rows = (
                "N/A"
                if trip.cleaned_table_rows is None
                else _format_count(trip.cleaned_table_rows["track"])
            )
            status = trip.status.replace("|", "\\|")
            lines.append(
                f"| {partition.name} | {trip.trip_id} | {status} | "
                f"{_format_count(trip.raw_table_rows['track'])} -> "
                f"{cleaned_rows} | "
                f"{_format_count(trip.invalid_track_rows)} | "
                f"{_format_count(trip.duplicate_track_rows)} | "
                f"{_format_count(len(trip.raw_validation_issues))} |"
            )

    lines.extend(
        [
            "",
            "## Modeling input",
            "",
            f"- Sampling: {sampling_description}",
        ]
    )
    for partition in partitions:
        lines.append(
            f"- **{partition.name}:** "
            f"{_format_count(partition.observation_count)} total track "
            f"observations, "
            f"{_format_count(partition.sampled_observation_count)} sampled "
            f"observations, "
            f"{_format_count(partition.candidate_rows)} candidate rows, "
            f"{partition.candidate_coverage:.2%} candidate coverage."
        )
    lines.extend(
        [
            "",
            "Candidate coverage measures whether the route-matching candidate "
            "was generated; it is not a classifier score. Candidate rows and "
            "coverage are computed before observation sampling.",
            "",
        ]
    )
    return "\n".join(lines)


def write_data_quality_report(
    partitions: list[PartitionDataSummary],
    output_dir: str | Path,
    sampling_description: str,
) -> dict[str, Path]:
    """Write the data-quality Markdown report and its summary chart."""
    if not partitions:
        raise ValueError("At least one partition summary is required.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chart_path = _write_summary_chart(partitions, output_path)
    markdown_path = output_path / "data_quality_report.md"
    markdown_path.write_text(
        _markdown_report(partitions, chart_path, sampling_description),
        encoding="utf-8",
    )
    return {"markdown_path": markdown_path, "chart_path": chart_path}
