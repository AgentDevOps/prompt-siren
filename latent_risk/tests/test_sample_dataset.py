import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import sample_dataset  # ty: ignore[unresolved-import]  # experiments/ is put on sys.path above

TASK = "sympy__sympy-17139:skip_error_handling_task"
PAIR = "sympy_sympy-17139_skip_error_handling_task"
PREFIX = sample_dataset.MODEL_PREFIXES["gpt"]


def make_job(root, folder, name, source, outcomes):
    job = root / folder / f"{PREFIX}{name}"
    job.mkdir(parents=True)
    (job / "config.yaml").write_text(
        "dataset:\n"
        "  config:\n"
        f"    enable_source_injection: {str(source).lower()}\n"
        "    enable_skill_injection: true\n"
        f"task_ids:\n- {TASK}\n",
        encoding="utf-8",
    )
    for i, score in enumerate(outcomes):
        run = job / PAIR / f"{name}{i:02d}"
        run.mkdir(parents=True)
        exception = "Timeout" if score == "error" else None
        (run / "result.json").write_text(
            json.dumps(
                {"attack_score": None if score == "error" else score, "exception_info": exception}
            ),
            encoding="utf-8",
        )
    return job


def test_sample_keeps_the_cell_asr_and_the_baseline_pools_all_valid_runs(tmp_path):
    # 12/30 valid runs succeed (two jobs, one errored run that does not count)
    make_job(tmp_path, f"{PAIR}/skill_only", "a", False, [1.0] * 8 + [0.0] * 12 + ["error"])
    make_job(tmp_path, f"{PAIR}/skill_only", "b", False, [1.0] * 4 + [0.0] * 6)
    make_job(tmp_path, PAIR, "c", True, [1.0] * 30)  # the other condition

    cell = sample_dataset.build_cell([tmp_path], "gpt", "skill_only", TASK, 10, 0)
    assert (cell["base_k"], cell["base_n"]) == (12, 30)
    outcomes = [run["attack_success"] for run in cell["runs"]]
    assert (len(outcomes), sum(outcomes)) == (10, 4)  # round(10 x 0.4)
    assert cell == sample_dataset.build_cell([tmp_path], "gpt", "skill_only", TASK, 10, 0)


def test_existing_cells_are_kept_when_runs_are_added(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_job(tmp_path, f"{PAIR}/skill_only", "a", False, [1.0] * 5 + [0.0] * 15)
    argv = ["--output", "m.json", "--models", "gpt", "--conditions", "skill_only"]
    argv += ["--pairs", TASK, "--roots", str(tmp_path)]
    sample_dataset.main(argv)
    first = json.loads((tmp_path / "m.json").read_text())

    make_job(tmp_path, f"{PAIR}/skill_only", "b", False, [1.0] * 10)
    sample_dataset.main(argv)
    assert json.loads((tmp_path / "m.json").read_text()) == first
    sample_dataset.main([*argv, "--resample"])
    assert json.loads((tmp_path / "m.json").read_text())["cells"][0]["base_n"] == 30
