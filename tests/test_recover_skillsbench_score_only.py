from pathlib import Path

import pytest

from scripts.recover_skillsbench_score_only import (
    SCRIPT_ALLOWLIST,
    extract_recorded_scripts,
    parse_file_editor_call,
)


def _event(title: str, line_number: int) -> dict:
    return {
        "type": "tool_call",
        "tool_call_id": f"call-{line_number}",
        "title": title,
        "_line_number": line_number,
    }


def test_parse_file_editor_call_accepts_recorded_ui_suffix() -> None:
    event = _event(
        'file_editor: {"command":"create","path":"/root/example.py",'
        '"file_text":"x = 1\\n"}: Editing /root/example.py',
        1,
    )

    assert parse_file_editor_call(event) == {
        "command": "create",
        "path": "/root/example.py",
        "file_text": "x = 1\n",
    }


def test_parse_file_editor_call_rejects_unexpected_suffix() -> None:
    event = _event(
        'file_editor: {"command":"view","path":"/root/example.py"} garbage',
        1,
    )

    with pytest.raises(ValueError, match="unexpected file_editor title suffix"):
        parse_file_editor_call(event)


def test_extract_recorded_scripts_replays_create_and_replace(tmp_path: Path) -> None:
    events = []
    line_number = 1
    for basename in sorted(SCRIPT_ALLOWLIST):
        events.append(
            _event(
                "file_editor: "
                f'{{"command":"create","path":"/root/{basename}",'
                '"file_text":"VALUE = 1\\n"}',
                line_number,
            )
        )
        line_number += 1
        events.append(
                _event(
                    "file_editor: "
                    f'{{"command":"str_replace","path":"/root/{basename}",'
                    '"old_str":"VALUE = 1","new_str":"VALUE = 2"}: '
                f"Editing /root/{basename}",
                line_number,
            )
        )
        line_number += 1

    evidence, operations = extract_recorded_scripts(events, tmp_path)

    assert len(operations) == len(SCRIPT_ALLOWLIST) * 2
    assert set(evidence) == SCRIPT_ALLOWLIST
    for basename in SCRIPT_ALLOWLIST:
        assert (tmp_path / basename).read_text(encoding="utf-8") == "VALUE = 2\n"
        assert evidence[basename]["size_bytes"] == (tmp_path / basename).stat().st_size


def test_extract_recorded_scripts_requires_unique_replace_target(
    tmp_path: Path,
) -> None:
    events = []
    for line_number, basename in enumerate(sorted(SCRIPT_ALLOWLIST), start=1):
        file_text = "x x\n" if basename == "build_workbook.py" else "x\n"
        events.append(
            _event(
                "file_editor: "
                f'{{"command":"create","path":"/root/{basename}",'
                f'"file_text":"{file_text.rstrip()}\\n"}}',
                line_number,
            )
        )
    events.append(
        _event(
            'file_editor: {"command":"str_replace",'
            '"path":"/root/build_workbook.py","old_str":"x",'
            '"new_str":"y"}',
            len(events) + 1,
        )
    )

    with pytest.raises(ValueError, match="expected one replacement target"):
        extract_recorded_scripts(events, tmp_path)
