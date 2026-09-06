"""Unit tests for Feature 3: Automated Dataset Cleaning & Sanity Repair Pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.data_clean import (
    CleanReport,
    clean_dataset,
    clean_row,
    is_echo_turn,
    repair_code_fences,
    repair_json_string,
    sanitize_text,
    strip_boilerplate,
)

runner = CliRunner()


def _create_sample_dirty_file(directory: Path) -> Path:
    """Create a temporary dirty JSONL file for CLI testing."""
    file_path = directory / "dirty_data.jsonl"
    data = [
        {
            "messages": [
                {"role": "user", "content": "How to add in Python?\u200b"},
                {
                    "role": "assistant",
                    "content": (
                        "Certainly! As an AI language model, here is the function:\n\n"
                        "```python\ndef add(a, b):\n    return a + b\n"
                    ),
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Empty question"},
                {"role": "assistant", "content": "   \n\u200b   "},
            ]
        },
    ]
    with open(file_path, "w", encoding="utf-8") as file_handle:
        for row in data:
            file_handle.write(json.dumps(row) + "\n")
    return file_path


def test_sanitize_text_control_characters():
    """Verify zero-width spaces, C0 controls, and CRLF are properly cleaned."""
    dirty_text = "Hello\u200b world\x00\x07!\r\nSecond line."
    cleaned, modified = sanitize_text(dirty_text)
    assert modified is True
    assert cleaned == "Hello world!\nSecond line."


def test_sanitize_text_clean_noop():
    """Verify clean text returns modified=False."""
    clean_text = "Standard clean text with\nnewlines and tabs\there."
    cleaned, modified = sanitize_text(clean_text)
    assert modified is False
    assert cleaned == clean_text


def test_repair_code_fences_unclosed():
    """Verify unclosed markdown code blocks are safely closed."""
    unclosed = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n"
    cleaned, modified = repair_code_fences(unclosed)
    assert modified is True
    assert cleaned.endswith("\n```")
    assert cleaned.count("```") == 2


def test_repair_code_fences_already_balanced():
    """Verify balanced code blocks are left intact."""
    balanced = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```"
    cleaned, modified = repair_code_fences(balanced)
    assert modified is False
    assert cleaned == balanced


def test_strip_boilerplate_prefixes():
    """Verify common LLM preamble is stripped."""
    text = "Certainly! As an AI language model, I can explain that. Quantum computing uses qubits."
    cleaned, modified = strip_boilerplate(text)
    assert modified is True
    assert cleaned == "Quantum computing uses qubits."


def test_strip_boilerplate_suffixes():
    """Verify common LLM sign-offs are stripped."""
    text = "Here is the answer. I hope this helps! Please let me know if you have any questions."
    cleaned, modified = strip_boilerplate(text)
    assert modified is True
    assert cleaned == "Here is the answer."


def test_repair_json_string_trailing_comma():
    """Verify trailing commas in JSON are repaired."""
    invalid_json = '{"name": "test", "value": 42,}'
    repaired, modified = repair_json_string(invalid_json)
    assert modified is True
    parsed = json.loads(repaired)
    assert parsed["name"] == "test"
    assert parsed["value"] == 42


def test_repair_json_string_markdown_wrapped():
    """Verify markdown-wrapped JSON is extracted cleanly."""
    wrapped_json = '```json\n{"action": "search", "query": "orders"}\n```'
    repaired, modified = repair_json_string(wrapped_json)
    assert modified is True
    parsed = json.loads(repaired)
    assert parsed["action"] == "search"
    assert parsed["query"] == "orders"


def test_is_echo_turn():
    """Verify echo turns where assistant repeats prompt verbatim are flagged."""
    assert is_echo_turn("What is your return policy?", "what is your return policy?") is True
    assert is_echo_turn("What is your return policy?", "Our return policy is 30 days.") is False


def test_clean_row_chatml():
    """Test cleaning a full ChatML row."""
    row = {
        "messages": [
            {"role": "user", "content": "How do I add in Python?\u200b"},
            {
                "role": "assistant",
                "content": (
                    "Sure! As an AI language model, here is the function:\n\n"
                    "```python\ndef add(a, b):\n    return a + b\n"
                ),
            },
        ]
    }
    cleaned, rules = clean_row(row, "chatml")
    assert cleaned is not None
    assert "Invisible & Control Chars" in rules
    assert "Boilerplate Disclaimers" in rules
    assert "Markdown Code Fence Repair" in rules

    assistant_msg = cleaned["messages"][1]["content"]
    assert "As an AI" not in assistant_msg
    assert assistant_msg.endswith("```")


def test_clean_row_alpaca():
    """Test cleaning an Alpaca row."""
    row = {
        "instruction": "Explain gravity\u200b",
        "input": "",
        "output": "Certainly! Gravity is a fundamental force. I hope this helps!",
    }
    cleaned, rules = clean_row(row, "alpaca")
    assert cleaned is not None
    assert "Boilerplate Disclaimers" in rules
    assert cleaned["output"] == "Gravity is a fundamental force."


def test_clean_row_sharegpt():
    """Test cleaning a ShareGPT row."""
    row = {
        "conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Sure! How can I help you today?"},
        ]
    }
    cleaned, rules = clean_row(row, "sharegpt")
    assert cleaned is not None
    assert cleaned["conversations"][1]["value"] == "How can I help you today?"


def test_clean_row_tool_calling():
    """Test repairing JSON in tool_calls arguments."""
    row = {
        "messages": [{"role": "user", "content": "Fetch order 123"}],
        "tool_calls": [
            {"name": "get_order", "arguments": '{"order_id": 123,}'}
        ],
    }
    cleaned, rules = clean_row(row, "tool-calling")
    assert cleaned is not None
    assert "Malformed JSON in Tools" in rules
    assert cleaned["tool_calls"][0]["arguments"] == '{"order_id": 123}'


def test_clean_row_empty_dropped():
    """Test that rows with empty assistant turns are dropped."""
    row = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "   \n\u200b   "},
        ]
    }
    cleaned, rules = clean_row(row, "chatml", min_tokens=1)
    assert cleaned is None
    assert "Empty / Whitespace Turns" in rules


def test_clean_row_echo_dropped():
    """Test that echo rows are dropped."""
    row = {
        "messages": [
            {"role": "user", "content": "Explain photosynthesis"},
            {"role": "assistant", "content": "Explain photosynthesis"},
        ]
    }
    cleaned, rules = clean_row(row, "chatml")
    assert cleaned is None
    assert "Target Leakage / Echo" in rules


def test_clean_dataset_batch():
    """Test batch dataset processing with report statistics."""
    data = [
        # Clean row
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        },
        # Row with unclosed code block
        {
            "messages": [
                {"role": "user", "content": "Code"},
                {"role": "assistant", "content": "```python\nprint(1)\n"},
            ]
        },
        # Empty row (should be dropped)
        {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": ""},
            ]
        },
    ]

    cleaned_data, report = clean_dataset(data, "chatml")
    assert isinstance(report, CleanReport)
    assert len(cleaned_data) == 2
    assert report.total_scanned == 3
    assert report.total_clean == 1
    assert report.total_modified == 1
    assert report.total_dropped == 1
    assert "Markdown Code Fence Repair" in report.rule_counts
    assert "Empty / Whitespace Turns" in report.rule_counts


def test_cli_clean_command_live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test full soup data clean CLI command."""
    monkeypatch.chdir(tmp_path)
    dirty_file = _create_sample_dirty_file(tmp_path)
    output_file = tmp_path / "cleaned_output.jsonl"

    result = runner.invoke(
        app,
        ["data", "clean", str(dirty_file), "-o", str(output_file)],
    )
    assert result.exit_code == 0
    assert output_file.exists()

    with open(output_file, encoding="utf-8") as file_handle:
        lines = [json.loads(line) for line in file_handle if line.strip()]

    # Out of 3 rows: 1 clean, 1 repaired, 1 empty dropped -> 2 output rows
    assert len(lines) == 2


def test_cli_clean_command_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test soup data clean with --dry-run flag (no file written)."""
    monkeypatch.chdir(tmp_path)
    dirty_file = _create_sample_dirty_file(tmp_path)
    output_file = tmp_path / "should_not_exist.jsonl"

    result = runner.invoke(
        app,
        ["data", "clean", str(dirty_file), "-o", str(output_file), "--dry-run"],
    )
    assert result.exit_code == 0
    assert not output_file.exists()


def test_cli_clean_command_json_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test soup data clean with --json flag."""
    monkeypatch.chdir(tmp_path)
    dirty_file = _create_sample_dirty_file(tmp_path)

    result = runner.invoke(
        app,
        ["data", "clean", str(dirty_file), "--dry-run", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["total_scanned"] == 3
    assert payload["output_rows"] == 2
    assert payload["total_dropped"] == 1
