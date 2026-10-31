"""High-fidelity Claude hook integration contract."""

import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from core.common import StateManager, get_timestamp_ms
from core.event_model import EventGraph, EventStatus, ToolEvent, TurnEvent
from tracing.claude_code.hooks.handlers import (
    _buffer_subagent,
    _DEFERRED_STOP_KEY,
    _QUEUED_PROMPTS_KEY,
    _QUEUED_TURN_PROMPT_ID,
    _cleanup_pending_subagents,
    _handle_post_tool_use,
    _handle_post_tool_use_failure,
    _handle_pre_tool_use,
    _handle_session_end,
    _handle_stop,
    _handle_stop_failure,
    _handle_subagent_start,
    _handle_subagent_stop,
    _handle_user_prompt_submit,
    _merge_pending_subagents,
    _merge_tool_observations,
    _pending_subagents,
    deferred_stop,
)
from tracing.claude_code.hooks.tool_buffer import ToolBuffer, ToolObservation
from tracing.claude_code.hooks.transcript import parse_claude_transcript

FIXTURE = Path(__file__).parent / "fixtures" / "main_tool_cycle.jsonl"
SUBAGENT_MAIN_FIXTURE = Path(__file__).parent / "fixtures" / "subagent_main.jsonl"
SUBAGENT_AGENT_FIXTURE = Path(__file__).parent / "fixtures" / "subagent_agent.jsonl"
ASYNC_MAIN_FIXTURE = Path(__file__).parent / "fixtures" / "async_subagent_main.jsonl"


def _spans(payload):
    return payload["resourceSpans"][0]["scopeSpans"][0]["spans"]


def _attrs(span):
    return {attribute["key"]: next(iter(attribute["value"].values())) for attribute in span["attributes"]}


def test_malformed_persisted_tool_timestamp_fails_soft_with_diagnostic():
    root = TurnEvent(
        event_id="turn",
        session_id="session",
        turn_id="1",
        sequence=0,
        started_at_ms=1,
        ended_at_ms=2,
        status=EventStatus.COMPLETED,
    )
    tool = ToolEvent(
        event_id="tool:call",
        parent_event_id="turn",
        session_id="session",
        turn_id="1",
        sequence=1,
        started_at_ms=1,
        ended_at_ms=None,
        status=EventStatus.PENDING,
        tool_call_id="call",
    )
    graph = EventGraph([root, tool])
    observation = ToolObservation.from_dict({"tool_use_id": "call", "ended_at_ms": "not-a-time", "status": "success"})

    assert _merge_tool_observations(graph, [observation]) == [observation]
    assert tool.ended_at_ms is None
    assert any(item.code == "invalid_timestamp" and item.event_id == tool.event_id for item in graph.diagnostics)


def test_hook_output_overlay_preserves_transcript_agent_correlation():
    root = TurnEvent(
        event_id="turn:test",
        session_id="session-agent-1",
        turn_id="1",
        sequence=0,
        started_at_ms=None,
        ended_at_ms=None,
        status=EventStatus.RUNNING,
    )
    graph = parse_claude_transcript(SUBAGENT_MAIN_FIXTURE, root)
    _merge_tool_observations(
        graph,
        [
            ToolObservation(
                tool_use_id="tool-agent-1",
                tool_response="hook-only agent result",
                status="success",
            )
        ],
    )

    agent_tool = next(event for event in graph.events if getattr(event, "tool_call_id", None) == "tool-agent-1")
    assert agent_tool.output["content"] == "hook-only agent result"
    assert agent_tool.output["toolUseResult"]["agentId"] == "agent-1"


def test_hook_overlay_is_first_wins_for_duplicate_tool_call_ids():
    tools = [
        ToolEvent(
            event_id=f"tool-{index}",
            parent_event_id=f"model-{index}",
            session_id="session",
            turn_id="1",
            sequence=index,
            started_at_ms=None,
            ended_at_ms=None,
            status=EventStatus.PENDING,
            tool_call_id="duplicate",
            tool_name="Read",
        )
        for index in (1, 2)
    ]
    observation = ToolObservation(tool_use_id="duplicate", tool_response="hook-result", status="success")

    matched = _merge_tool_observations(EventGraph(tools), [observation])

    assert matched == [observation]
    assert tools[0].status is EventStatus.COMPLETED
    assert tools[0].output == "hook-result"
    assert tools[1].status is EventStatus.PENDING
    assert tools[1].output is None


def test_unmatched_snapshots_are_not_acknowledged(tmp_path: Path):
    state = StateManager(tmp_path, state_file=tmp_path / "state.json", lock_path=tmp_path / "state.lock")
    state.init_state()
    descriptor = {
        "agent_id": "unmatched",
        "agent_type": "general-purpose",
        "transcript_path": str(SUBAGENT_AGENT_FIXTURE),
    }
    state.set("pending_subagents", json.dumps({"unmatched": descriptor}))
    root = TurnEvent(
        event_id="turn",
        session_id="session",
        turn_id="1",
        sequence=0,
        started_at_ms=None,
        ended_at_ms=None,
        status=EventStatus.COMPLETED,
    )

    matched = _merge_pending_subagents(EventGraph([root]), _pending_subagents(state))
    _cleanup_pending_subagents(state, matched)

    assert matched == {}
    assert "unmatched" in _pending_subagents(state)


def test_parallel_subagent_buffering_is_atomic(tmp_path: Path):
    state = StateManager(tmp_path, state_file=tmp_path / "state.json", lock_path=tmp_path / "state.lock")
    state.init_state()
    count = 24
    for index in range(count):
        state.set(f"subagent_agent-{index}_start_time", str(index))

    def add(index: int) -> None:
        _buffer_subagent(
            state,
            {
                "agent_id": f"agent-{index}",
                "agent_type": "general-purpose",
                "transcript_path": str(SUBAGENT_MAIN_FIXTURE),
                "agent_transcript_path": str(SUBAGENT_AGENT_FIXTURE),
            },
            ended_at_ms=100 + index,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(add, range(count)))

    assert set(_pending_subagents(state)) == {f"agent-{index}" for index in range(count)}


def test_pending_subagent_cleanup_only_acknowledges_unchanged_snapshot(tmp_path: Path):
    state = StateManager(tmp_path, state_file=tmp_path / "state.json", lock_path=tmp_path / "state.lock")
    state.init_state()
    state.set("subagent_old_start_time", "1")
    _buffer_subagent(
        state,
        {
            "agent_id": "old",
            "agent_type": "general-purpose",
            "transcript_path": str(SUBAGENT_MAIN_FIXTURE),
            "agent_transcript_path": str(SUBAGENT_AGENT_FIXTURE),
        },
        ended_at_ms=10,
    )
    snapshot = _pending_subagents(state)

    state.set("subagent_new_start_time", "2")
    _buffer_subagent(
        state,
        {
            "agent_id": "new",
            "agent_type": "general-purpose",
            "transcript_path": str(SUBAGENT_MAIN_FIXTURE),
            "agent_transcript_path": str(SUBAGENT_AGENT_FIXTURE),
        },
        ended_at_ms=20,
    )

    _cleanup_pending_subagents(state, snapshot)

    assert list(_pending_subagents(state)) == ["new"]
    assert state.get("subagent_new_start_time") == "2"


def test_real_v2_transcript_is_exported_once_as_correlated_tree(tmp_path: Path):
    state_file = tmp_path / "state.json"
    state = StateManager(tmp_path, state_file=state_file, lock_path=tmp_path / "state.lock")
    state.init_state()
    state.set("session_id", "session-main-1")
    state.set("current_trace_id", "a" * 32)
    state.set("current_trace_span_id", "b" * 16)
    state.set("current_trace_start_time", "1767268800000")
    state.set("current_trace_prompt", "synthetic prompt")
    state.set("trace_count", "1")
    state.set("trace_start_line", "0")
    state.set("project_name", "synthetic-project")
    state.set(
        "vcs_attrs",
        json.dumps({"vcs.ref.head.revision": "c0ffee", "vcs.ref.head.name": "v20", "harness.dirty": True}),
    )
    sent = []

    def send_success(payload):
        sent.append(payload)
        return True

    common_input = {"session_id": "session-main-1", "transcript_path": str(FIXTURE)}
    with (
        mock.patch("tracing.claude_code.hooks.handlers.resolve_session", return_value=state),
        mock.patch("tracing.claude_code.hooks.handlers.resolve_transcript_path", return_value=FIXTURE),
        mock.patch(
            "tracing.claude_code.hooks.handlers.get_timestamp_ms",
            side_effect=[1767268801000, 1767268802000, 1767268803000, 1767268804000, 1767268805000],
        ),
        mock.patch("tracing.claude_code.hooks.handlers.send_span", side_effect=send_success),
    ):
        _handle_pre_tool_use(
            {
                **common_input,
                "tool_name": "Read",
                "tool_use_id": "tool-read-1",
                "tool_input": {"file_path": "/synthetic/project/hello.py"},
            }
        )
        _handle_post_tool_use(
            {
                **common_input,
                "tool_name": "Read",
                "tool_use_id": "tool-read-1",
                "tool_input": {"file_path": "/synthetic/project/hello.py"},
                "tool_response": "hook read output",
            }
        )
        _handle_stop({**common_input, "last_assistant_message": "SYNTHETIC_TOOL_OK"})

    assert len(sent) == 1
    spans = _spans(sent[0])
    assert [span["name"] for span in spans] == [
        "Turn 1",
        "LLM call 1: qwen3-coder-next",
        "Read",
        "LLM call 2: qwen3-coder-next",
        "Bash",
        "LLM call 3: qwen3-coder-next",
    ]
    assert spans[0]["spanId"] == "b" * 16
    assert "parentSpanId" not in spans[0]
    assert spans[2]["parentSpanId"] == spans[1]["spanId"]
    assert spans[4]["parentSpanId"] == spans[3]["spanId"]
    assert _attrs(spans[0])["openinference.span.kind"] == "CHAIN"
    assert "llm.token_count.total" not in _attrs(spans[0])
    assert _attrs(spans[0])["vcs.ref.head.revision"] == "c0ffee"
    assert _attrs(spans[0])["vcs.ref.head.name"] == "v20"
    assert _attrs(spans[0])["harness.dirty"] is True
    assert "vcs.ref.head.revision" not in _attrs(spans[1])
    assert _attrs(spans[2])["output.value"] == "hook read output"
    assert _attrs(spans[2])["tool.call.id"] == "tool-read-1"
    assert ToolBuffer(state).all() == []
    assert state.get("current_trace_id") is None


def test_legacy_post_tool_without_transcript_keeps_immediate_fallback(tmp_path: Path):
    state_file = tmp_path / "state.json"
    state = StateManager(tmp_path, state_file=state_file, lock_path=tmp_path / "state.lock")
    state.init_state()
    state.set("session_id", "legacy-session")
    state.set("current_trace_id", "trace-legacy")
    state.set("current_trace_span_id", "span-legacy")
    sent = []

    with (
        mock.patch("tracing.claude_code.hooks.handlers.resolve_session", return_value=state),
        mock.patch("tracing.claude_code.hooks.handlers.send_span", side_effect=sent.append),
    ):
        _handle_post_tool_use(
            {
                "tool_name": "Read",
                "tool_use_id": "legacy-tool",
                "tool_input": {"file_path": "/legacy.py"},
                "tool_response": "legacy output",
            }
        )

    assert len(sent) == 1
    assert _spans(sent[0])[0]["name"] == "Read"


def test_failed_export_retains_buffer_and_reuses_span_ids_on_retry(tmp_path: Path):
    state = StateManager(tmp_path, state_file=tmp_path / "state.json", lock_path=tmp_path / "state.lock")
    state.init_state()
    for key, value in {
        "session_id": "session-main-1",
        "current_trace_id": "a" * 32,
        "current_trace_span_id": "b" * 16,
        "current_trace_start_time": "1767268800000",
        "current_trace_prompt": "synthetic prompt",
        "trace_count": "1",
        "trace_start_line": "0",
        "project_name": "synthetic-project",
    }.items():
        state.set(key, value)

    common_input = {"session_id": "session-main-1", "transcript_path": str(FIXTURE)}
    attempts = []

    def send_with_retry(payload):
        attempts.append(payload)
        return len(attempts) == 2

    with (
        mock.patch("tracing.claude_code.hooks.handlers.resolve_session", return_value=state),
        mock.patch("tracing.claude_code.hooks.handlers.resolve_transcript_path", return_value=FIXTURE),
        mock.patch(
            "tracing.claude_code.hooks.handlers.get_timestamp_ms",
            side_effect=[1767268801000, 1767268802000, 1767268803000, 1767268804000],
        ),
        mock.patch("tracing.claude_code.hooks.handlers.send_span", side_effect=send_with_retry),
    ):
        _handle_pre_tool_use(
            {
                **common_input,
                "tool_name": "Read",
                "tool_use_id": "tool-read-1",
                "tool_input": {"file_path": "/synthetic/project/hello.py"},
            }
        )
        _handle_post_tool_use(
            {
                **common_input,
                "tool_name": "Read",
                "tool_use_id": "tool-read-1",
                "tool_response": "hook read output",
            }
        )
        _handle_stop({**common_input, "last_assistant_message": "SYNTHETIC_TOOL_OK"})

        assert state.get("current_trace_id") == "a" * 32
        assert len(ToolBuffer(state).all()) == 1
        assert state.get("high_fidelity_span_ids") is not None

        _handle_stop({**common_input, "last_assistant_message": "SYNTHETIC_TOOL_OK"})

    assert len(attempts) == 2
    assert [span["spanId"] for span in _spans(attempts[0])] == [span["spanId"] for span in _spans(attempts[1])]
    assert ToolBuffer(state).all() == []
    assert state.get("current_trace_id") is None
    assert state.get("high_fidelity_span_ids") is None


def test_foreground_subagent_is_buffered_and_exported_inside_main_tree(tmp_path: Path):
    state = StateManager(tmp_path, state_file=tmp_path / "state.json", lock_path=tmp_path / "state.lock")
    state.init_state()
    for key, value in {
        "session_id": "session-agent-1",
        "current_trace_id": "c" * 32,
        "current_trace_span_id": "d" * 16,
        "current_trace_start_time": "1767272400000",
        "current_trace_prompt": "Delegate one read-only subagent.",
        "trace_count": "1",
        "trace_start_line": "0",
        "project_name": "synthetic-project",
    }.items():
        state.set(key, value)

    sent = []
    export_results = iter([False, True])

    def send_with_retry(payload):
        sent.append(payload)
        return next(export_results)

    with (
        mock.patch("tracing.claude_code.hooks.handlers.resolve_session", return_value=state),
        mock.patch("tracing.claude_code.hooks.handlers.resolve_transcript_path", return_value=SUBAGENT_MAIN_FIXTURE),
        mock.patch(
            "tracing.claude_code.hooks.handlers.get_timestamp_ms",
            side_effect=[1767272401000, 1767272403000, 1767272404000, 1767272405000],
        ),
        mock.patch("tracing.claude_code.hooks.handlers.send_span", side_effect=send_with_retry),
    ):
        _handle_subagent_start(
            {
                "session_id": "session-agent-1",
                "transcript_path": str(SUBAGENT_MAIN_FIXTURE),
                "agent_id": "agent-1",
                "agent_type": "synthetic-explorer",
                "prompt_id": "prompt-agent-1",
            }
        )
        _handle_subagent_stop(
            {
                "session_id": "session-agent-1",
                "transcript_path": str(SUBAGENT_MAIN_FIXTURE),
                "agent_transcript_path": str(SUBAGENT_AGENT_FIXTURE),
                "agent_id": "agent-1",
                "agent_type": "synthetic-explorer",
                "last_assistant_message": "Function: greeting; return: SYNTHETIC_TOOL_OK",
            }
        )

        assert sent == []
        assert state.get("pending_subagents") is not None

        _handle_stop(
            {
                "session_id": "session-agent-1",
                "transcript_path": str(SUBAGENT_MAIN_FIXTURE),
                "last_assistant_message": "SUBAGENT_SCHEMA_OK",
            }
        )
        assert len(sent) == 1
        assert state.get("pending_subagents") is not None

        _handle_stop(
            {
                "session_id": "session-agent-1",
                "transcript_path": str(SUBAGENT_MAIN_FIXTURE),
                "last_assistant_message": "SUBAGENT_SCHEMA_OK",
            }
        )

    assert len(sent) == 2
    assert [span["spanId"] for span in _spans(sent[0])] == [span["spanId"] for span in _spans(sent[1])]
    spans = _spans(sent[1])
    assert [span["name"] for span in spans] == [
        "Turn 1",
        "LLM call 1: qwen3-coder-next",
        "Agent",
        "Subagent: synthetic-explorer",
        "LLM call 2: qwen3-coder-next",
        "Read",
        "LLM call 3: qwen3-coder-next",
        "LLM call 4: qwen3-coder-next",
    ]
    assert spans[2]["parentSpanId"] == spans[1]["spanId"]
    assert spans[3]["parentSpanId"] == spans[2]["spanId"]
    assert spans[4]["parentSpanId"] == spans[3]["spanId"]
    assert spans[5]["parentSpanId"] == spans[4]["spanId"]
    assert spans[6]["parentSpanId"] == spans[3]["spanId"]
    assert spans[7]["parentSpanId"] == spans[0]["spanId"]
    assert _attrs(spans[3])["openinference.span.kind"] == "AGENT"
    assert _attrs(spans[3])["subagent.id"] == "agent-1"
    assert _attrs(spans[3])["subagent.type"] == "synthetic-explorer"
    assert state.get("pending_subagents") is None
    assert state.get("subagent_agent-1_start_time") is None


def test_background_subagent_is_exported_under_its_launching_agent_span(tmp_path: Path):
    state = StateManager(tmp_path, state_file=tmp_path / "state.json", lock_path=tmp_path / "state.lock")
    state.init_state()
    for key, value in {
        "session_id": "session-async-1",
        "current_trace_id": "e" * 32,
        "current_trace_span_id": "f" * 16,
        "current_trace_start_time": "1767272400000",
        "current_trace_prompt": "Delegate one read-only subagent in the background.",
        "trace_count": "1",
        "trace_start_line": "0",
        "project_name": "synthetic-project",
        "user_id": "dev@example.com",
    }.items():
        state.set(key, value)

    sent = []
    clock = iter(range(1767272401000, 1767272500000, 1000))

    def send(payload):
        sent.append(payload)
        return True

    with (
        mock.patch("tracing.claude_code.hooks.handlers.resolve_session", return_value=state),
        mock.patch("tracing.claude_code.hooks.handlers.resolve_transcript_path", return_value=ASYNC_MAIN_FIXTURE),
        mock.patch("tracing.claude_code.hooks.handlers.get_timestamp_ms", side_effect=lambda: next(clock)),
        mock.patch("tracing.claude_code.hooks.handlers.send_span", side_effect=send),
    ):
        _handle_subagent_start(
            {
                "session_id": "session-async-1",
                "transcript_path": str(ASYNC_MAIN_FIXTURE),
                "agent_id": "agent-1",
                "agent_type": "synthetic-explorer",
            }
        )
        # The launching turn ends while the subagent is still running.
        _handle_stop(
            {
                "session_id": "session-async-1",
                "transcript_path": str(ASYNC_MAIN_FIXTURE),
                "last_assistant_message": "LAUNCHED_IN_BACKGROUND",
            }
        )
        assert len(sent) == 1
        turn_spans = _spans(sent[0])
        assert [span["name"] for span in turn_spans] == [
            "Turn 1",
            "LLM call 1: qwen3-coder-next",
            "Agent",
            "LLM call 2: qwen3-coder-next",
        ]
        assert state.get("current_trace_id") is None
        assert state.get("async_subagents") is not None

        # It stops later, with no turn open.
        _handle_subagent_stop(
            {
                "session_id": "session-async-1",
                "transcript_path": str(ASYNC_MAIN_FIXTURE),
                "agent_transcript_path": str(SUBAGENT_AGENT_FIXTURE),
                "agent_id": "agent-1",
                "agent_type": "synthetic-explorer",
                "last_assistant_message": "Function: greeting; return: SYNTHETIC_TOOL_OK",
            }
        )

    assert len(sent) == 2
    agent_span = turn_spans[2]
    spans = _spans(sent[1])
    assert [span["name"] for span in spans] == [
        "Subagent: synthetic-explorer",
        "LLM call 1: qwen3-coder-next",
        "Read",
        "LLM call 2: qwen3-coder-next",
    ]
    assert {span["traceId"] for span in spans} == {"e" * 32}
    assert spans[0]["parentSpanId"] == agent_span["spanId"]
    assert spans[1]["parentSpanId"] == spans[0]["spanId"]
    assert spans[2]["parentSpanId"] == spans[1]["spanId"]
    assert spans[3]["parentSpanId"] == spans[0]["spanId"]
    subagent = _attrs(spans[0])
    assert subagent["openinference.span.kind"] == "AGENT"
    assert subagent["subagent.id"] == "agent-1"
    assert subagent["user.id"] == "dev@example.com"
    # Real duration: from SubagentStart to SubagentStop, not the launch call.
    assert int(spans[0]["endTimeUnixNano"]) > int(spans[0]["startTimeUnixNano"])
    assert int(spans[0]["startTimeUnixNano"]) == 1767272401000 * 1_000_000
    assert _attrs(spans[2])["subagent.id"] == "agent-1"
    assert state.get("async_subagents") is None
    assert state.get("subagent_agent-1_start_time") is None


def test_background_subagent_that_stops_inside_its_turn_keeps_the_foreground_path(tmp_path: Path):
    state = StateManager(tmp_path, state_file=tmp_path / "state.json", lock_path=tmp_path / "state.lock")
    state.init_state()
    for key, value in {
        "session_id": "session-async-1",
        "current_trace_id": "e" * 32,
        "current_trace_span_id": "f" * 16,
        "current_trace_start_time": "1767272400000",
        "trace_count": "1",
        "trace_start_line": "0",
    }.items():
        state.set(key, value)

    sent = []
    clock = iter(range(1767272401000, 1767272500000, 1000))
    with (
        mock.patch("tracing.claude_code.hooks.handlers.resolve_session", return_value=state),
        mock.patch("tracing.claude_code.hooks.handlers.resolve_transcript_path", return_value=ASYNC_MAIN_FIXTURE),
        mock.patch("tracing.claude_code.hooks.handlers.get_timestamp_ms", side_effect=lambda: next(clock)),
        mock.patch("tracing.claude_code.hooks.handlers.send_span", side_effect=lambda p: sent.append(p) or True),
    ):
        _handle_subagent_stop(
            {
                "session_id": "session-async-1",
                "transcript_path": str(ASYNC_MAIN_FIXTURE),
                "agent_transcript_path": str(SUBAGENT_AGENT_FIXTURE),
                "agent_id": "agent-1",
                "agent_type": "synthetic-explorer",
            }
        )
        _handle_stop({"session_id": "session-async-1", "transcript_path": str(ASYNC_MAIN_FIXTURE)})

    assert len(sent) == 1
    names = [span["name"] for span in _spans(sent[0])]
    assert names.index("Subagent: synthetic-explorer") == names.index("Agent") + 1
    assert state.get("async_subagents") is None
    assert state.get("pending_subagents") is None


def test_failed_tool_hook_is_correlated_and_marks_tool_span_error(tmp_path: Path):
    state = StateManager(tmp_path, state_file=tmp_path / "state.json", lock_path=tmp_path / "state.lock")
    state.init_state()
    for key, value in {
        "session_id": "session-main-1",
        "current_trace_id": "e" * 32,
        "current_trace_span_id": "f" * 16,
        "current_trace_start_time": "1767268800000",
        "current_trace_prompt": "synthetic prompt",
        "trace_count": "1",
        "trace_start_line": "0",
    }.items():
        state.set(key, value)

    sent = []

    def send_success(payload):
        sent.append(payload)
        return True

    common_input = {"session_id": "session-main-1", "transcript_path": str(FIXTURE)}
    with (
        mock.patch("tracing.claude_code.hooks.handlers.resolve_session", return_value=state),
        mock.patch("tracing.claude_code.hooks.handlers.resolve_transcript_path", return_value=FIXTURE),
        mock.patch(
            "tracing.claude_code.hooks.handlers.get_timestamp_ms",
            side_effect=[1767268801000, 1767268802000, 1767268803000],
        ),
        mock.patch("tracing.claude_code.hooks.handlers.send_span", side_effect=send_success),
    ):
        _handle_pre_tool_use(
            {
                **common_input,
                "tool_name": "Read",
                "tool_use_id": "tool-read-1",
                "tool_input": {"file_path": "/synthetic/project/missing.py"},
            }
        )
        _handle_post_tool_use_failure(
            {
                **common_input,
                "tool_name": "Read",
                "tool_use_id": "tool-read-1",
                "tool_input": {"file_path": "/synthetic/project/missing.py"},
                "error": "synthetic read failure",
            }
        )
        assert sent == []
        _handle_stop({**common_input, "last_assistant_message": "handled"})

    tool = next(span for span in _spans(sent[0]) if _attrs(span).get("tool.call.id") == "tool-read-1")
    assert tool["status"]["code"] == 2
    assert _attrs(tool)["error.message"] == "synthetic read failure"
    assert ToolBuffer(state).all() == []


FIXTURE_LINES = FIXTURE.read_text(encoding="utf-8").splitlines(keepends=True)
FINAL_STOP = {"session_id": "session-main-1", "hook_event_name": "Stop", "last_assistant_message": "SYNTHETIC_TOOL_OK"}
STOPPED_AT_MS = 1767268805000  # after every record of the fixture
WORKER_INPUT = dict(FINAL_STOP, arize_deferred_stop=True, arize_stopped_at_ms=STOPPED_AT_MS)


def _open_turn(tmp_path: Path, lines: "list[str]") -> "tuple[StateManager, Path]":
    """A session whose turn 1 is open over a transcript holding *lines*."""
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("".join(lines), encoding="utf-8")
    state = StateManager(tmp_path, state_file=tmp_path / "state.json", lock_path=tmp_path / "state.lock")
    state.init_state()
    for key, value in {
        "session_id": "session-main-1",
        "current_trace_id": "a" * 32,
        "current_trace_span_id": "b" * 16,
        "current_trace_start_time": "1767268800000",
        "current_trace_prompt": "synthetic prompt",
        "trace_count": "1",
        "trace_start_line": "0",
    }.items():
        state.set(key, value)
    return state, transcript


def _hooks(state: StateManager, transcript: Path, *, sleep=None, popen=None, stdin=None, monotonic=None):
    """Patch the session, transcript, sender, clock, worker spawner and stdin of the handlers."""
    sent, sleep_mock, popen_mock = [], mock.Mock(side_effect=sleep), popen or mock.Mock()
    patches = {
        "resolve_session": {"return_value": state},
        "resolve_transcript_path": {"return_value": transcript},
        "ensure_session_initialized": {},
        "send_span": {"side_effect": lambda payload: sent.append(payload) or True},
        "time.sleep": {"new": sleep_mock},
        "subprocess.Popen": {"new": popen_mock},
        "_read_stdin": {"return_value": stdin or {}},
        "gc_stale_state_files": {},
    }
    if monotonic is not None:
        patches["time.monotonic"] = {"side_effect": monotonic}
    stack = ExitStack()
    for target, kwargs in patches.items():
        stack.enter_context(mock.patch(f"tracing.claude_code.hooks.handlers.{target}", **kwargs))
    return stack, sent, sleep_mock, popen_mock


def _defer_marker_until(state: StateManager, deadline_ms: int, then=None):
    """Set a deferred-Stop deadline; return a sleep side effect that runs *then* and finishes the worker."""
    state.set(_DEFERRED_STOP_KEY, str(deadline_ms))

    def worker_finishes(_seconds):
        if then:
            then()
        state.delete("current_trace_id")
        state.delete(_DEFERRED_STOP_KEY)

    return worker_finishes


def test_stop_exports_at_once_when_the_final_record_is_written(tmp_path: Path):
    state, transcript = _open_turn(tmp_path, FIXTURE_LINES)
    stack, sent, sleep, popen = _hooks(state, transcript)
    with stack:
        _handle_stop(dict(FINAL_STOP))

    popen.assert_not_called()
    sleep.assert_not_called()
    assert [span["name"] for span in _spans(sent[0])][-1] == "LLM call 3: qwen3-coder-next"
    assert state.get("current_trace_id") is None


def test_stop_hands_the_turn_to_a_detached_worker_when_the_final_record_is_missing(tmp_path: Path):
    state, transcript = _open_turn(tmp_path, FIXTURE_LINES[:-1])
    stack, sent, sleep, popen = _hooks(state, transcript)
    with stack:
        _handle_stop(dict(FINAL_STOP))

    assert sent == []
    sleep.assert_not_called()
    assert state.get("current_trace_id") == "a" * 32
    assert int(state.get(_DEFERRED_STOP_KEY)) > get_timestamp_ms()
    (argv,), kwargs = popen.call_args
    assert argv[0] == sys.executable
    assert kwargs["stdout"] is subprocess.DEVNULL and kwargs["stderr"] is subprocess.DEVNULL
    assert kwargs.get("start_new_session") or kwargs.get("creationflags")
    payload = json.loads(popen.return_value.stdin.write.call_args.args[0])
    assert payload["arize_deferred_stop"] is True
    assert isinstance(payload["arize_stopped_at_ms"], int)
    popen.return_value.stdin.close.assert_called_once()


def test_worker_command_imports_its_entry_point(tmp_path: Path):
    state, transcript = _open_turn(tmp_path, FIXTURE_LINES[:-1])
    stack, _sent, _sleep, popen = _hooks(state, transcript)
    with stack:
        _handle_stop(dict(FINAL_STOP))
    (argv,), _kwargs = popen.call_args

    # The hook launchers put the plugin root on PYTHONPATH; here the repo root plays that part.
    repo_root = str(Path(__file__).resolve().parents[3])
    result = subprocess.run(
        argv, input=b"{}", capture_output=True, cwd=tmp_path, timeout=60, env={**os.environ, "PYTHONPATH": repo_root}
    )

    assert result.returncode == 0, result.stderr
    assert b"deferred stop failed" not in result.stderr


def test_worker_exports_the_turn_once_the_final_record_lands(tmp_path: Path):
    state, transcript = _open_turn(tmp_path, FIXTURE_LINES[:-1])
    state.set(_DEFERRED_STOP_KEY, str(get_timestamp_ms() + 60_000))

    def claude_code_writes_the_final_record(_seconds):
        with open(transcript, "a", encoding="utf-8") as f:
            f.write(FIXTURE_LINES[-1])

    stack, sent, sleep, popen = _hooks(state, transcript, sleep=claude_code_writes_the_final_record, stdin=WORKER_INPUT)
    with stack:
        deferred_stop()

    assert sleep.call_count == 1
    popen.assert_not_called()
    spans = _spans(sent[0])
    assert spans[-1]["name"] == "LLM call 3: qwen3-coder-next"
    root = next(span for span in spans if not span.get("parentSpanId"))
    assert int(root["endTimeUnixNano"]) == STOPPED_AT_MS * 1_000_000
    assert state.get("current_trace_id") is None
    assert state.get(_DEFERRED_STOP_KEY) is None


def test_worker_gives_up_after_the_timeout_and_sends_the_legacy_turn(tmp_path: Path):
    # A text-only turn whose final record never lands: only the prompt is written.
    state, transcript = _open_turn(tmp_path, FIXTURE_LINES[:1])
    state.set(_DEFERRED_STOP_KEY, str(get_timestamp_ms() + 60_000))
    stack, sent, sleep, _popen = _hooks(state, transcript, stdin=WORKER_INPUT, monotonic=itertools.count(0, 0.25))
    with stack:
        deferred_stop()

    assert sleep.call_count == 19  # reads at 0.25 s ticks until the 5 s deadline, sleeping between them
    span = _spans(sent[0])[0]
    assert span["name"] == "Turn 1"
    assert _attrs(span)["openinference.span.kind"] == "LLM"
    assert int(span["endTimeUnixNano"]) == STOPPED_AT_MS * 1_000_000
    assert state.get(_DEFERRED_STOP_KEY) is None


def test_stop_exports_at_once_when_the_worker_cannot_start(tmp_path: Path):
    state, transcript = _open_turn(tmp_path, FIXTURE_LINES[:-1])
    stack, sent, _sleep, _popen = _hooks(state, transcript, popen=mock.Mock(side_effect=OSError("no fork")))
    with stack:
        _handle_stop(dict(FINAL_STOP))

    assert len(sent) == 1
    assert state.get(_DEFERRED_STOP_KEY) is None
    assert state.get("current_trace_id") is None


def test_repeated_stop_waits_for_the_worker_instead_of_exporting_again(tmp_path: Path):
    state, transcript = _open_turn(tmp_path, FIXTURE_LINES[:-1])
    worker_finishes = _defer_marker_until(state, get_timestamp_ms() + 60_000)
    stack, sent, sleep, popen = _hooks(state, transcript, sleep=worker_finishes)
    with stack:
        _handle_stop(dict(FINAL_STOP))

    assert sleep.call_count == 1
    assert sent == []
    popen.assert_not_called()


def test_next_prompt_waits_for_the_worker_before_the_fail_safe(tmp_path: Path):
    state, transcript = _open_turn(tmp_path, FIXTURE_LINES)
    worker_finishes = _defer_marker_until(state, get_timestamp_ms() + 60_000)
    stack, sent, sleep, _popen = _hooks(state, transcript, sleep=worker_finishes)
    with stack:
        _handle_user_prompt_submit({"session_id": "session-main-1", "prompt": "next prompt"})

    assert sleep.call_count == 1
    assert sent == []  # the worker exported the turn, so the fail-safe had nothing to close
    assert state.get("current_trace_id") not in (None, "a" * 32)
    assert state.get("current_trace_prompt") == "next prompt"


def test_next_prompt_ignores_a_worker_past_its_deadline(tmp_path: Path):
    state, transcript = _open_turn(tmp_path, FIXTURE_LINES)
    state.set(_DEFERRED_STOP_KEY, str(get_timestamp_ms() - 1))
    stack, sent, sleep, _popen = _hooks(state, transcript)
    with stack:
        _handle_user_prompt_submit({"session_id": "session-main-1", "prompt": "next prompt"})

    sleep.assert_not_called()
    assert len(sent) == 1  # the fail-safe closed the turn the worker left open


def test_session_end_waits_for_the_worker_before_removing_the_state(tmp_path: Path):
    state, transcript = _open_turn(tmp_path, FIXTURE_LINES)

    def state_still_there():
        assert state.state_file.exists()

    worker_finishes = _defer_marker_until(state, get_timestamp_ms() + 60_000, then=state_still_there)
    stack, _sent, sleep, _popen = _hooks(state, transcript, sleep=worker_finishes)
    with stack, mock.patch("tracing.claude_code.hooks.handlers.error"):
        _handle_session_end({"session_id": "session-main-1"})

    assert sleep.call_count == 1
    assert not state.state_file.exists()


def test_fail_safe_close_neither_waits_nor_defers(tmp_path: Path):
    state, transcript = _open_turn(tmp_path, FIXTURE_LINES[:-1])
    fail_safe = {
        "session_id": "session-main-1",
        "hook_event_name": "UserPromptSubmit",
        "last_assistant_message": "(Turn closed by fail-safe: Stop hook did not fire)",
    }
    stack, sent, sleep, popen = _hooks(state, transcript)
    with stack:
        _handle_stop(fail_safe)

    sleep.assert_not_called()
    popen.assert_not_called()
    assert len(sent) == 1


def _user_line(text: str) -> str:
    return json.dumps({"type": "user", "message": {"role": "user", "content": text}}) + "\n"


def _queued_command_line(text: str) -> str:
    return json.dumps({"type": "attachment", "attachment": {"type": "queued_command", "prompt": text}}) + "\n"


def _queued(state: StateManager) -> "list[str]":
    return json.loads(state.get(_QUEUED_PROMPTS_KEY) or "[]")


def test_prompt_with_the_running_turns_prompt_id_is_queued_not_closed(tmp_path: Path):
    state, transcript = _open_turn(tmp_path, FIXTURE_LINES[:-1])
    state.set("current_trace_prompt_id", "prompt-1")
    stack, sent, _sleep, _popen = _hooks(state, transcript)
    with stack:
        _handle_user_prompt_submit({"session_id": "session-main-1", "prompt": "di dos", "prompt_id": "prompt-1"})

    assert sent == []
    assert state.get("current_trace_id") == "a" * 32
    assert state.get("current_trace_prompt") == "synthetic prompt"
    assert _queued(state) == ["di dos"]


def test_prompt_with_a_new_prompt_id_closes_the_orphan_and_drops_its_queue(tmp_path: Path):
    # A turn interrupted with Esc gets no Stop; the next prompt carries a new prompt_id.
    state, transcript = _open_turn(tmp_path, FIXTURE_LINES[:-1])
    state.set("current_trace_prompt_id", "prompt-1")
    state.set(_QUEUED_PROMPTS_KEY, json.dumps(["queued behind the orphan"]))
    stack, sent, _sleep, _popen = _hooks(state, transcript)
    with stack:
        _handle_user_prompt_submit({"session_id": "session-main-1", "prompt": "di tres", "prompt_id": "prompt-2"})

    assert len(sent) == 1
    assert "fail-safe" in _attrs(_spans(sent[0])[0])["output.value"]
    assert state.get("current_trace_id") not in (None, "a" * 32)
    assert state.get("current_trace_prompt") == "di tres"
    assert state.get("current_trace_prompt_id") == "prompt-2"
    assert _queued(state) == []


def test_stop_opens_the_next_turn_for_prompts_queued_after_it(tmp_path: Path):
    # A text-only turn: Claude Code runs the queued prompts as a new turn after Stop.
    state, transcript = _open_turn(tmp_path, [*FIXTURE_LINES, _user_line("di dos"), _user_line("di adiós")])
    state.set("current_trace_prompt_id", "prompt-1")
    state.set(_QUEUED_PROMPTS_KEY, json.dumps(["di dos", "di adiós"]))
    stack, sent, _sleep, _popen = _hooks(state, transcript)
    with stack:
        _handle_stop(dict(FINAL_STOP))

    assert [span["name"] for span in _spans(sent[0])][-1] == "LLM call 3: qwen3-coder-next"
    assert state.get("current_trace_id") not in (None, "a" * 32)
    assert state.get("current_trace_prompt") == "di dos\n\ndi adiós"
    assert state.get("current_trace_prompt_id") == _QUEUED_TURN_PROMPT_ID
    assert state.get("trace_start_line") == str(len(FIXTURE_LINES))
    assert state.get("trace_count") == "2"
    assert int(state.get("current_trace_start_time")) <= get_timestamp_ms()
    assert _queued(state) == []


def test_stop_leaves_prompts_taken_mid_turn_in_the_ended_turn(tmp_path: Path):
    # A turn with tool calls takes queued prompts between calls.
    lines = [*FIXTURE_LINES[:-1], _queued_command_line("di dos"), FIXTURE_LINES[-1]]
    state, transcript = _open_turn(tmp_path, lines)
    state.set("current_trace_prompt_id", "prompt-1")
    state.set(_QUEUED_PROMPTS_KEY, json.dumps(["di dos"]))
    stack, sent, _sleep, _popen = _hooks(state, transcript)
    with stack:
        _handle_stop(dict(FINAL_STOP))

    assert len(sent) == 1
    assert state.get("current_trace_id") is None
    assert _queued(state) == []


def test_prompt_submitted_during_a_turn_opened_from_the_queue_is_queued(tmp_path: Path):
    state, transcript = _open_turn(tmp_path, FIXTURE_LINES[:-1])
    state.set("current_trace_prompt_id", _QUEUED_TURN_PROMPT_ID)
    stack, sent, _sleep, _popen = _hooks(state, transcript)
    with stack:
        _handle_user_prompt_submit({"session_id": "session-main-1", "prompt": "di más", "prompt_id": "prompt-9"})

    assert sent == []
    assert state.get("current_trace_id") == "a" * 32
    assert _queued(state) == ["di más"]


def test_deferred_worker_opens_the_next_turn_after_the_final_record(tmp_path: Path):
    state, transcript = _open_turn(tmp_path, FIXTURE_LINES[:-1])
    state.set("current_trace_prompt_id", "prompt-1")
    state.set(_QUEUED_PROMPTS_KEY, json.dumps(["di dos"]))
    state.set(_DEFERRED_STOP_KEY, str(get_timestamp_ms() + 60_000))

    def claude_code_ends_the_turn_and_dequeues(_seconds):
        with open(transcript, "a", encoding="utf-8") as f:
            f.write(FIXTURE_LINES[-1] + _user_line("di dos"))

    stack, sent, _sleep, _popen = _hooks(
        state, transcript, sleep=claude_code_ends_the_turn_and_dequeues, stdin=WORKER_INPUT
    )
    with stack:
        deferred_stop()

    assert [span["name"] for span in _spans(sent[0])][-1] == "LLM call 3: qwen3-coder-next"
    assert state.get("current_trace_prompt") == "di dos"
    assert state.get("trace_start_line") == str(len(FIXTURE_LINES))
    assert state.get("current_trace_start_time") == str(STOPPED_AT_MS)


def test_failed_turn_opens_the_next_turn_for_its_queue(tmp_path: Path):
    # Claude Code writes the dequeued prompts after the StopFailure hook.
    state, transcript = _open_turn(tmp_path, FIXTURE_LINES[:-1])
    state.set("current_trace_prompt_id", "prompt-1")
    state.set(_QUEUED_PROMPTS_KEY, json.dumps(["di dos"]))
    stack, sent, _sleep, _popen = _hooks(state, transcript)
    with stack:
        _handle_stop_failure({"session_id": "session-main-1", "hook_event_name": "StopFailure", "error": "overloaded"})

    assert len(sent) == 1
    assert state.get("current_trace_prompt") == "di dos"


def test_fail_safe_close_does_not_open_a_turn_for_the_queue(tmp_path: Path):
    state, transcript = _open_turn(tmp_path, FIXTURE_LINES)
    state.set(_QUEUED_PROMPTS_KEY, json.dumps(["di dos"]))
    fail_safe = {
        "session_id": "session-main-1",
        "hook_event_name": "UserPromptSubmit",
        "last_assistant_message": "(Turn closed by fail-safe: Stop hook did not fire)",
    }
    stack, sent, _sleep, _popen = _hooks(state, transcript)
    with stack:
        _handle_stop(fail_safe)

    assert len(sent) == 1
    assert state.get("current_trace_id") is None
    assert _queued(state) == ["di dos"]
