from pathlib import Path

from scripts.score_skillsbench_artifact_only import parse_pytest_log, tree_evidence


def test_parse_pytest_log_handles_failed_summary_lines_without_suffix() -> None:
    log = """
collecting ... collected 3 items
E   AssertionError: first failure
E   AssertionError: second failure
PASSED ../verifier/test_outputs.py::test_ok
FAILED ../verifier/test_outputs.py::test_first - AssertionErr...
FAILED ../verifier/test_outputs.py::test_second
"""

    summary = parse_pytest_log(log)

    assert summary["collected"] == 3
    assert summary["passed_node_ids"] == [
        "../verifier/test_outputs.py::test_ok"
    ]
    assert summary["failed_node_ids"] == [
        "../verifier/test_outputs.py::test_first",
        "../verifier/test_outputs.py::test_second",
    ]
    assert summary["failure_details"] == [
        {
            "node_id": "../verifier/test_outputs.py::test_first",
            "assertion": "first failure",
        },
        {
            "node_id": "../verifier/test_outputs.py::test_second",
            "assertion": "second failure",
        },
    ]


def test_tree_evidence_is_content_addressed_and_stable(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested" / "b.txt").write_text("beta", encoding="utf-8")

    first = tree_evidence(tmp_path)
    second = tree_evidence(tmp_path)

    assert first == second
    assert first["file_count"] == 2
    assert set(first["files"]) == {"a.txt", "nested/b.txt"}
    assert len(first["snapshot_sha256"]) == 64
