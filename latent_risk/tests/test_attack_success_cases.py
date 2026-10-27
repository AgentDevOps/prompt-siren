from pathlib import Path

from latent_risk.attack_success_cases import record_attack_success_case


def test_record_attack_success_case_converts_legacy_rows(tmp_path: Path):
    path = tmp_path / "attack_success_cases.md"
    path.write_text(
        "task                                              model\n"
        "sympy__sympy-15345:modify_readme_curl_bash_task   gpt-5.1\n"
    )

    added = record_attack_success_case(
        task_id="sympy__sympy-15345:config_injection_exfil_task",
        model="gpt-5.1",
        attack_score=1.0,
        job_name="job-1",
        run_id="abc12345",
        path=path,
    )

    assert added is True
    content = path.read_text()
    assert "| Task | Model | Attack score | Job | Run ID |" in content
    assert "| sympy__sympy-15345:modify_readme_curl_bash_task | gpt-5.1 | - | - | - |" in content
    assert (
        "| sympy__sympy-15345:config_injection_exfil_task | gpt-5.1 | 1.0 | job-1 | abc12345 |"
        in content
    )


def test_record_attack_success_case_deduplicates_by_task_and_model(tmp_path: Path):
    path = tmp_path / "attack_success_cases.md"

    first_added = record_attack_success_case(
        task_id="task1:malicious",
        model="gpt-5.1",
        attack_score=1.0,
        job_name="job-1",
        run_id="abc12345",
        path=path,
    )
    second_added = record_attack_success_case(
        task_id="task1:malicious",
        model="gpt-5.1",
        attack_score=1.0,
        job_name="job-2",
        run_id="def67890",
        path=path,
    )

    assert first_added is True
    assert second_added is False
    assert path.read_text().count("task1:malicious") == 1
