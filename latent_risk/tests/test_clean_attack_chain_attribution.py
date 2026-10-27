import json

import pytest

from scripts import clean_attack_chain_attribution as clean


def write_run(
    tmp_path,
    name,
    *,
    status="ok",
    attack_score=None,
    write_summary=False,
):
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "execution.json").write_text(json.dumps({"messages": [], "run_id": name}))
    (run_dir / "attack_chain_attribution.json").write_text(json.dumps({"status": status}))
    (run_dir / "attack_chain_attribution.md").write_text(f"# {name}\n")
    if attack_score is not None:
        (run_dir / "result.json").write_text(json.dumps({"attack_score": attack_score}))
    if write_summary:
        (run_dir / "attack_chain_attribution_summary.md").write_text("summary\n")
    return run_dir


def test_dry_run_lists_but_never_deletes(tmp_path, capsys):
    write_run(tmp_path, "a", status="ok")
    write_run(tmp_path, "b", status="failed")

    assert clean.main([str(tmp_path)]) == 0
    output = capsys.readouterr().out

    assert "would_remove" in output
    assert (tmp_path / "a" / "attack_chain_attribution.json").exists()
    assert (tmp_path / "b" / "attack_chain_attribution.json").exists()
    assert (tmp_path / "a" / "attack_chain_attribution.md").exists()


def test_apply_deletes_json_and_markdown_pair(tmp_path):
    write_run(tmp_path, "a", status="ok")

    assert clean.main([str(tmp_path), "--apply"]) == 0

    assert not (tmp_path / "a" / "attack_chain_attribution.json").exists()
    assert not (tmp_path / "a" / "attack_chain_attribution.md").exists()
    assert (tmp_path / "a" / "execution.json").exists()  # never touches source files


def test_status_filter_only_removes_matching_status(tmp_path):
    write_run(tmp_path, "a", status="ok")
    write_run(tmp_path, "b", status="failed")
    write_run(tmp_path, "c", status="no_candidate")

    clean.main([str(tmp_path), "--status", "failed", "--apply"])

    assert (tmp_path / "a" / "attack_chain_attribution.json").exists()
    assert not (tmp_path / "b" / "attack_chain_attribution.json").exists()
    assert (tmp_path / "c" / "attack_chain_attribution.json").exists()


def test_status_filter_accepts_multiple_values(tmp_path):
    write_run(tmp_path, "a", status="failed")
    write_run(tmp_path, "b", status="insufficient_context")
    write_run(tmp_path, "c", status="ok")

    clean.main([str(tmp_path), "--status", "failed", "--status", "insufficient_context", "--apply"])

    assert not (tmp_path / "a" / "attack_chain_attribution.json").exists()
    assert not (tmp_path / "b" / "attack_chain_attribution.json").exists()
    assert (tmp_path / "c" / "attack_chain_attribution.json").exists()


def test_only_succeeded_attacks_uses_result_json_not_prompt_derived_status(tmp_path):
    write_run(tmp_path, "succeeded", status="ok", attack_score=1.0)
    write_run(tmp_path, "failed", status="ok", attack_score=0.0)
    write_run(tmp_path, "unknown", status="ok", attack_score=None)

    clean.main([str(tmp_path), "--only-succeeded-attacks", "--apply"])

    assert not (tmp_path / "succeeded" / "attack_chain_attribution.json").exists()
    assert (tmp_path / "failed" / "attack_chain_attribution.json").exists()
    assert (tmp_path / "unknown" / "attack_chain_attribution.json").exists()


def test_only_failed_attacks_excludes_unknown_outcome(tmp_path):
    write_run(tmp_path, "succeeded", status="ok", attack_score=1.0)
    write_run(tmp_path, "failed", status="ok", attack_score=0.0)
    write_run(tmp_path, "unknown", status="ok", attack_score=None)

    clean.main([str(tmp_path), "--only-failed-attacks", "--apply"])

    assert (tmp_path / "succeeded" / "attack_chain_attribution.json").exists()
    assert not (tmp_path / "failed" / "attack_chain_attribution.json").exists()
    assert (tmp_path / "unknown" / "attack_chain_attribution.json").exists()  # never guessed


def test_status_and_outcome_filters_combine(tmp_path):
    write_run(tmp_path, "a", status="failed", attack_score=1.0)
    write_run(tmp_path, "b", status="failed", attack_score=0.0)
    write_run(tmp_path, "c", status="ok", attack_score=0.0)

    clean.main([str(tmp_path), "--status", "failed", "--only-failed-attacks", "--apply"])

    assert (tmp_path / "a" / "attack_chain_attribution.json").exists()
    assert not (tmp_path / "b" / "attack_chain_attribution.json").exists()
    assert (tmp_path / "c" / "attack_chain_attribution.json").exists()


def test_summary_is_removed_by_default_but_can_be_kept(tmp_path):
    write_run(tmp_path, "a", status="ok", write_summary=True)

    removals = clean.plan_removals(
        [tmp_path], statuses=None, outcome_filter="any", include_summaries=True
    )
    assert any(path.name == "attack_chain_attribution_summary.md" for path in removals)

    clean.main([str(tmp_path), "--no-summary", "--apply"])
    assert (tmp_path / "a" / "attack_chain_attribution_summary.md").exists()
    assert not (tmp_path / "a" / "attack_chain_attribution.json").exists()


def test_never_touches_chain_or_execution_files(tmp_path):
    run_dir = write_run(tmp_path, "a", status="ok")
    (run_dir / "attack_chain_judge.json").write_text("{}")
    (run_dir / "attack_chain_labeled.json").write_text("{}")

    clean.main([str(tmp_path), "--apply"])

    assert (run_dir / "attack_chain_judge.json").exists()
    assert (run_dir / "attack_chain_labeled.json").exists()
    assert (run_dir / "execution.json").exists()


def test_malformed_attribution_json_is_skipped_not_crashed(tmp_path, capsys):
    run_dir = tmp_path / "broken"
    run_dir.mkdir()
    (run_dir / "attack_chain_attribution.json").write_text("not json")

    result = clean.main([str(tmp_path), "--status", "failed"])

    assert result == 0
    assert "warning" in capsys.readouterr().err


def test_no_matches_reports_cleanly(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert clean.main([str(tmp_path)]) == 0
    assert "No matching attribution output found." in capsys.readouterr().out


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        clean.main([str(tmp_path / "does-not-exist")])
