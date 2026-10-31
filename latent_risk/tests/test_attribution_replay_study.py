import json

from scripts import attribution_replay_study as study

TASK = "sympy__sympy-17139:skip_error_handling_task"
PAIR = "sympy_sympy-17139_skip_error_handling_task"


def make_job(root, folder, name, source, successes, valid):
    job = root / folder / f"{study.JOB_PREFIX}{name}"
    job.mkdir(parents=True)
    (job / "config.yaml").write_text(
        "dataset:\n"
        "  config:\n"
        f"    enable_source_injection: {str(source).lower()}\n"
        "    enable_skill_injection: true\n"
        "    use_cache: false\n"
        f"task_ids:\n- {TASK}\n",
        encoding="utf-8",
    )
    (job / "asr.json").write_text(
        json.dumps({"attack_successes": successes, "valid_attack_runs": valid}), encoding="utf-8"
    )
    return job


def test_discover_jobs_selects_condition_and_pools_pair_baseline(tmp_path):
    standard = make_job(tmp_path, PAIR, "a", True, 3, 30)
    small = make_job(tmp_path, f"{PAIR}/skill_only", "b", False, 1, 10)
    large = make_job(tmp_path, f"{PAIR}/skill_only", "c", False, 6, 20)

    jobs, _ = study.discover_jobs(tmp_path, 0.8)
    assert [(job.path, job.base_k, job.base_n) for job in jobs] == [(standard, 3, 30)]

    jobs, _ = study.discover_jobs(tmp_path, 0.8, "skill_only")
    assert [(job.path, job.base_k, job.base_n) for job in jobs] == [(small, 1, 10), (large, 6, 20)]

    jobs, _ = study.discover_jobs(tmp_path, 0.8, "skill_only", pool_baseline=True)
    assert [(job.base_k, job.base_n) for job in jobs] == [(7, 30), (7, 30)]


def test_pooled_baseline_is_what_the_ceiling_applies_to(tmp_path):
    make_job(tmp_path, f"{PAIR}/skill_only", "b", False, 9, 10)
    make_job(tmp_path, f"{PAIR}/skill_only", "c", False, 6, 20)

    jobs, _ = study.discover_jobs(tmp_path, 0.8, "skill_only")
    assert [job.path.name[-1] for job in jobs] == ["c"]

    jobs, _ = study.discover_jobs(tmp_path, 0.8, "skill_only", pool_baseline=True)
    assert len(jobs) == 2  # 15/30 pooled is below the ceiling


def test_reuse_images_sets_use_cache_in_either_config_format(tmp_path):
    for text in ('    "use_cache": false\n', "    use_cache: false\n"):
        (tmp_path / "config.yaml").write_text(f"dataset:\n  config:\n{text}", encoding="utf-8")
        study.reuse_images(tmp_path)
        patched = (tmp_path / "config.yaml").read_text(encoding="utf-8")
        assert patched == f"dataset:\n  config:\n{text.replace('false', 'true')}"
