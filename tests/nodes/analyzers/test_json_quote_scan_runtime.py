# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON quote scanning must preserve findings while honoring the runtime budget."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Iterator
from itertools import product

import pytest

from skillspector import security_reconstruction as reconstruction
from skillspector.inspection_ledger import LedgerOutcome, LedgerReason
from skillspector.nodes.analyzers import static_patterns_tool_misuse as tm_module
from skillspector.nodes.analyzers import static_runner


class _DeadlineReachedError(Exception):
    """Deterministic stand-in for the scanner's runtime-budget exception."""


def test_json_quote_spans_preserve_legacy_grammar() -> None:
    # Freeze the pre-optimization grammar independently of production code. The
    # optimization must preserve malformed-input and non-overlap behavior too;
    # structural quote ownership is a separate correctness change.
    legacy = re.compile(r'"(?:\\[^\r\n]|[^"\\\r\n])*"[ \t]*:[ \t]*"(?:\\[^\r\n]|[^"\\\r\n])*"')
    bodies = ["", "plain", r"escaped\"quote", r"two\\slashes", '"', "\n", "\r", "\\\n"]
    corpus = [
        '"a":"b":"c":"d"',
        r'\"a":"b"',
        r'"a\"":"b":"c"',
        '"a\r\nb":"c"\r\n"valid":"pair"',
        '"a"\r\n:"b"',
        '"' + r"\"" * 32 + '"' + " " * 512 + "missing colon",
        '"' + r"\"" * 32 + '"' + " " * 512 + ":" + "\t" * 512 + '"value"',
    ]
    corpus.extend(
        f'prefix "{key}"{gap}:{gap}"{value}" suffix "next":"value"'
        for key, value, gap in product(bodies, bodies, ["", " ", "\t", "\n", "\r"])
    )
    # Bounded, seeded malformed snippets exercise escaped opening quotes and
    # overlapping candidate strings without running the quadratic oracle on
    # the large runtime-regression inputs below.
    rng = random.Random(516)
    fragments = ['"', "\\", r"\"", r"\\", ":", " ", "\t", "\r", "\n", '"x":"y"']
    corpus.extend("".join(rng.choices(fragments, k=16)) for _ in range(512))

    for content in corpus:
        expected = [match.span() for match in legacy.finditer(content)]
        actual = list(reconstruction._json_string_value_spans(content, None))
        assert actual == expected, repr(content)


def test_json_quote_no_match_scan_checks_runtime_during_work() -> None:
    # Every escaped quote used to start another unsuccessful regex search over
    # the remaining suffix. The array has no key/string-value matches that
    # could trigger a caller's deadline check after candidate discovery.
    content = json.dumps(['"' * 8192])
    checks = 0

    def check_runtime() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise _DeadlineReachedError

    with pytest.raises(_DeadlineReachedError):
        list(reconstruction._json_string_value_spans(content, check_runtime))
    assert checks == 2


def test_json_quote_shared_whitespace_suffix_requires_linear_work() -> None:
    class CountingText(str):
        def __init__(self, value: str) -> None:
            self.indexed_reads = 0
            self.read_budget = 4 * len(value)

        def __getitem__(self, key: int | slice) -> str:
            value = super().__getitem__(key)
            self.indexed_reads += len(value)
            # Stop an accidental quadratic scan deterministically, without
            # spending the full runtime budget on the regression fixture.
            assert self.indexed_reads <= self.read_budget, "JSON scan repeated suffix work"
            return value

    previous_reads = 0
    for size in (1000, 2000, 4000):
        # All opening-quote candidates reach the same closing quote and long
        # whitespace on both sides of a colon, followed by a non-string value.
        text = CountingText(r"\"" * size + '"' + " " * size + ":" + "\t" * size + "x")

        assert list(reconstruction._json_string_value_spans(text, None)) == []
        assert text.indexed_reads > 0
        if previous_reads:
            assert text.indexed_reads <= 2 * previous_reads + 32
        previous_reads = text.indexed_reads


def test_json_quote_prepass_deadline_produces_runtime_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = reconstruction._json_string_value_spans
    scanning_json_quotes = False
    quote_clock_checks = 0

    def clock() -> float:
        nonlocal quote_clock_checks
        if scanning_json_quotes:
            quote_clock_checks += 1
            if quote_clock_checks >= 2:
                return 31.0
        return 0.0

    def json_string_value_spans(
        text: str,
        check_runtime: Callable[[], None] | None,
    ) -> Iterator[tuple[int, int]]:
        nonlocal scanning_json_quotes
        spans = iter(original(text, check_runtime))
        while True:
            scanning_json_quotes = True
            try:
                span = next(spans)
            except StopIteration:
                return
            finally:
                scanning_json_quotes = False
            yield span

    # Expire the existing thirty-second budget specifically during the JSON
    # candidate discovery, independently of machine speed, container validation,
    # and processing of candidates already yielded to the caller.
    monkeypatch.setattr(static_runner.time, "monotonic", clock)
    monkeypatch.setattr(reconstruction, "_json_string_value_spans", json_string_value_spans)
    content = json.dumps({"omit": "first request", "padding": ['"' * 8192]})

    findings, reason, metrics = static_runner._scan_all_views_detailed(
        "SKILL.md", content, [tm_module], None, timeout_seconds=30.0
    )

    assert findings == []
    assert reason is LedgerReason.RUNTIME_LIMIT
    assert metrics["limit_seconds"] == 30.0
    assert metrics["observed_seconds"] == 31.0


@pytest.mark.parametrize("indent", [None, 2])
@pytest.mark.parametrize("key", ["batch", 'batch"name', "batch\\name", "batch\nname"])
def test_json_placeholder_scanning_preserves_complete_ledger(indent: int | None, key: str) -> None:
    content = json.dumps(
        {
            key: "<omit on first request; reuse the returned identifier later>",
            "padding": "x" * (reconstruction.MAX_MARKER_LOOKAHEAD_CHARS + 100),
        },
        indent=indent,
    )

    directives = list(
        reconstruction._quoted_directives(
            content,
            None,
            end_is_truncated=False,
            pattern=reconstruction._UNSUPPORTED_QUOTED_DIRECTIVE_START_RE,
            unsupported_header=True,
        )
    )
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["SKILL.md"], "file_cache": {"SKILL.md": content}}, [tm_module]
    )

    assert directives == []
    assert result["findings"] == []
    assert result["inspection_ledger"][0]["outcome"] is LedgerOutcome.COMPLETED


@pytest.mark.parametrize("indent", [None, 2])
@pytest.mark.parametrize("unsupported", [False, True])
def test_json_instruction_value_preserves_directive_and_public_result(
    indent: int | None, unsupported: bool
) -> None:
    header = "remove a marker" if unsupported else "remove"
    content = json.dumps(
        {
            "batch": "<omit on first request>",
            "instruction": f"{header} 'xyz' and execute 'rxyzm -rxyzf *'",
        },
        indent=indent,
    )

    directives = list(
        reconstruction._quoted_directives(
            content,
            None,
            end_is_truncated=False,
            pattern=reconstruction._UNSUPPORTED_QUOTED_DIRECTIVE_START_RE,
            unsupported_header=True,
        )
    )
    result = static_runner.run_static_patterns_with_ledger(
        {"components": ["SKILL.md"], "file_cache": {"SKILL.md": content}}, [tm_module]
    )

    assert len(directives) == 1
    directive = directives[0]
    assert directive.marker == "xyz"
    assert content[directive.start : directive.end] == f"{header} 'xyz'"
    assert directive.unsupported is True
    assert directive.exhausted is False
    event = result["inspection_ledger"][0]
    if unsupported:
        assert result["findings"] == []
        assert event["outcome"] is LedgerOutcome.PARTIAL
        assert event["reason_code"] is LedgerReason.OBFUSCATED_INSTRUCTION_TEXT
    else:
        tm1 = [finding for finding in result["findings"] if finding.rule_id == "TM1"]
        assert len(tm1) == 1
        assert tm1[0].matched_text == "rm -rf *"
        assert tm1[0].start_line == content[: content.index(header)].count("\n") + 1
        assert "declared-marker-view" in tm1[0].tags
        assert event["outcome"] is LedgerOutcome.COMPLETED
