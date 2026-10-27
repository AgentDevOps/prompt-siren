# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Maintain the docs table of successful attack examples."""

from __future__ import annotations

from pathlib import Path

from filelock import FileLock

ATTACK_SUCCESS_CASES_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "attack_success_cases.md"
)

_COLUMNS = ("task", "model", "attack_score", "job", "run_id")
_HEADERS = {
    "task": "Task",
    "model": "Model",
    "attack_score": "Attack score",
    "job": "Job",
    "run_id": "Run ID",
}
_HEADER_ALIASES = {
    "task": "task",
    "model": "model",
    "attack score": "attack_score",
    "attack_score": "attack_score",
    "job": "job",
    "run id": "run_id",
    "run_id": "run_id",
}


def record_attack_success_case(
    *,
    task_id: str,
    model: str,
    attack_score: float,
    job_name: str,
    run_id: str,
    path: Path = ATTACK_SUCCESS_CASES_PATH,
) -> bool:
    """Add a successful attack case to the docs table if it is not already present.

    Existing records are considered the same case when both task and model match.
    Returns True when a new row was added.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(f"{path.suffix}.lock")

    with FileLock(lock_path):
        existing_rows = _read_rows(path)
        key = (task_id, model)
        if any((row.get("task"), row.get("model")) == key for row in existing_rows):
            _write_rows(path, existing_rows)
            return False

        existing_rows.append(
            {
                "task": task_id,
                "model": model,
                "attack_score": f"{attack_score:.1f}",
                "job": job_name,
                "run_id": run_id,
            }
        )
        _write_rows(path, existing_rows)
        return True


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    lines = path.read_text().splitlines()
    table_rows = _parse_markdown_table(lines)
    if table_rows:
        return table_rows

    return _parse_legacy_rows(lines)


def _parse_markdown_table(lines: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    headers: list[str] | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(set(cell.replace(":", "").strip()) <= {"-"} for cell in cells):
            continue
        if headers is None:
            headers = [_HEADER_ALIASES.get(cell.lower(), cell.lower()) for cell in cells]
            continue

        row = dict.fromkeys(_COLUMNS, "-")
        for header, cell in zip(headers, cells, strict=False):
            if header in _COLUMNS:
                row[header] = _unescape_cell(cell)
        if row["task"] != "-" and row["model"] != "-":
            rows.append(row)

    return rows


def _parse_legacy_rows(lines: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.lower().split() == ["task", "model"]:
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        rows.append(
            {
                "task": parts[0],
                "model": parts[1],
                "attack_score": "-",
                "job": "-",
                "run_id": "-",
            }
        )
    return rows


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    rendered = [
        "# Attack Success Cases",
        "",
        "| Task | Model | Attack score | Job | Run ID |",
        "| --- | --- | ---: | --- | --- |",
    ]

    seen: set[tuple[str, str]] = set()
    for row in rows:
        task = row.get("task", "-")
        model = row.get("model", "-")
        key = (task, model)
        if key in seen:
            continue
        seen.add(key)
        rendered.append(
            "| "
            + " | ".join(_escape_cell(row.get(column, "-")) for column in _COLUMNS)
            + " |"
        )

    rendered.append("")
    path.write_text("\n".join(rendered))


def _escape_cell(value: str) -> str:
    text = str(value or "-").replace("\n", "<br>")
    return text.replace("|", r"\|")


def _unescape_cell(value: str) -> str:
    return value.replace(r"\|", "|") or "-"
