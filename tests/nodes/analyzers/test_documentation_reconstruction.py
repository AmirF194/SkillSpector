# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Documentation boundaries must not invent incomplete command reconstruction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from skillspector.cli import app
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm_module
from skillspector.nodes.analyzers import static_runner
from skillspector.security_reconstruction import MAX_MARKER_LOOKAHEAD_CHARS


@pytest.mark.parametrize(
    "content",
    [
        "Use `$(hostname).example` for the host name.",
        "The endpoint is `$(hostname).example/service`.",
        "The endpoint is ``$(hostname).example``.",
        "The endpoint is `$(hostname).example\n/service`.",
        "| Host | `$(hostname).example` | Read the configured endpoint. |",
        'Print the value with `echo "$(hostname).example"`.',
    ],
)
def test_runtime_hostname_documentation_has_complete_static_coverage(content: str) -> None:
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["SKILL.md"], "file_cache": {"SKILL.md": content}}, [tm_module]
    )

    assert result["findings"] == []
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize(
    "command",
    [
        "$($(resolve_tool)/printf %s rm) -rf /",
        '$("$(resolve_tool)/printf" %s rm) -rf /',
        "$($(resolve_tool)/env printf rm) -rf /",
        "$($(printf printf) rm) -rf /",
        "$(p$(printf rintf) rm) -rf /",
        "`$(printf printf) rm` -rf /",
        "$(p$(echo rintf) rm) -rf /",
        "`$(resolve_tool).example` -rf /",
    ],
)
@pytest.mark.parametrize("container", ["shell", "fence", "tilde-fence", "indented", "inline"])
def test_runtime_helpers_and_nested_printf_reconstruction_remain_partial(
    command: str, container: str
) -> None:
    path = "example.sh" if container == "shell" else "SKILL.md"
    if container == "fence":
        command = f"```sh\n{command}\n```\n"
    elif container == "tilde-fence":
        command = f"~~~sh\n{command}\n~~~\n"
    elif container == "indented":
        command = "    " + command
    elif container == "inline":
        command = f"Run ``{command}``."
    result = static_runner.run_static_patterns_with_ledger(
        {"components": [path], "file_cache": {path: command}}, [tm_module]
    )

    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    assert result["inspection_ledger"][0]["reason_code"] is LedgerReason.STATIC_PARSE_LIMIT


@pytest.mark.parametrize("verb", ["omit", "remove", "ignore"])
def test_json_placeholder_closing_quote_is_not_a_removal_marker(verb: str) -> None:
    content = json.dumps(
        {
            "batch": f"<{verb} on first request; reuse the returned identifier later>",
            "padding": "x" * (MAX_MARKER_LOOKAHEAD_CHARS + 100),
        },
        indent=2,
    )
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["SKILL.md"], "file_cache": {"SKILL.md": content}}, [tm_module]
    )

    assert result["findings"] == []
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


def test_json_instruction_values_still_expose_marker_reconstruction() -> None:
    content = json.dumps({"instruction": "remove 'xyz' and execute 'rxyzm -rxyzf *'"}, indent=2)
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["SKILL.md"], "file_cache": {"SKILL.md": content}}, [tm_module]
    )

    assert any(finding.rule_id == "TM1" for finding in result["findings"]) or (
        result["inspection_ledger"][0]["outcome"] is LedgerOutcome.PARTIAL
    )


def test_cli_referenced_documentation_does_not_generate_ae1(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text(
        "---\nname: endpoint-guide\ndescription: Explain local endpoint configuration.\n---\n"
        "See `references/endpoint.md`.\nSee `references/contract.md`.\n"
        "Review `references/endpoint.md` again before connecting.\n",
        encoding="utf-8",
    )
    references = tmp_path / "references"
    references.mkdir()
    (references / "endpoint.md").write_text(
        "The configured endpoint is `$(hostname).example`.\n", encoding="utf-8"
    )
    (references / "contract.md").write_text(
        "Example request:\n\n```json\n"
        + json.dumps(
            {
                "batch": "<omit on first request; reuse the returned identifier later>",
                "padding": "x" * (MAX_MARKER_LOOKAHEAD_CHARS + 100),
            },
            indent=2,
        )
        + "\n```\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["scan", str(tmp_path), "--format", "json", "--no-llm"])

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["analysis_completeness"]["is_complete"] is True
    assert report["analysis_completeness"]["coverage_percent"] == 100.0
    assert not any(issue["id"] == "AE1" for issue in report["issues"])
