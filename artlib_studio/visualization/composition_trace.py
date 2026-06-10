"""Non-UI views over composition event traces."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Union

from ..composition.events import CompositionEvent, CompositionEventType

TraceEvent = Union[CompositionEvent, Dict[str, Any]]


def _event_dict(event: TraceEvent) -> Dict[str, Any]:
    if isinstance(event, CompositionEvent):
        return event.to_dict()
    return dict(event)


def build_module_timeline(trace: Iterable[TraceEvent]) -> List[Dict[str, Any]]:
    """Build chronological module-event rows, excluding graph-only events."""
    rows = []
    for position, event in enumerate(trace):
        data = _event_dict(event)
        if data.get("module_id") is None:
            continue
        rows.append(
            {
                "position": position,
                "sample_index": data.get(
                    "external_sample_index", data.get("step_index")
                ),
                "module_id": data["module_id"],
                "event_type": data["type"],
                "payload": data.get("payload", {}),
            }
        )
    return rows


def build_signal_flow_table(trace: Iterable[TraceEvent]) -> List[Dict[str, Any]]:
    """Build rows describing signals propagated across graph edges."""
    rows = []
    for event in trace:
        data = _event_dict(event)
        if data.get("type") != CompositionEventType.SIGNAL_SENT.value:
            continue
        payload = data.get("payload", {})
        rows.append(
            {
                "sample_index": data.get(
                    "external_sample_index", data.get("step_index")
                ),
                "source_module_id": data.get("module_id"),
                "target_module_id": payload.get("target_module_id"),
                "edge_type": payload.get("edge_type"),
                "source_signal_type": payload.get("source_signal_type"),
                "signal_type": payload.get("signal_type"),
                "signal_payload": payload.get("signal_payload", {}),
                "transform_name": payload.get("transform_name"),
            }
        )
    return rows


def summarize_graph_step(
    trace: Iterable[TraceEvent],
    sample_index: int,
) -> Dict[str, Any]:
    """Summarize module selections and signal flow for one external sample."""
    matching = [
        _event_dict(event)
        for event in trace
        if _event_dict(event).get(
            "external_sample_index", _event_dict(event).get("step_index")
        )
        == sample_index
    ]
    selected_categories = {
        event["module_id"]: event.get("payload", {}).get("category_id")
        for event in matching
        if event.get("type") == CompositionEventType.MODULE_SELECTED_CATEGORY.value
    }
    return {
        "sample_index": sample_index,
        "selected_categories": selected_categories,
        "signals": build_signal_flow_table(matching),
        "events": matching,
        "settled": any(
            event.get("type") == CompositionEventType.GRAPH_SETTLED.value
            for event in matching
        ),
    }


def build_expectation_table(trace: Iterable[TraceEvent]) -> List[Dict[str, Any]]:
    """Build rows for top-down expectations and their evaluation."""
    expectation_types = {
        CompositionEventType.EXPECTATION_SENT.value,
        CompositionEventType.EXPECTATION_RECEIVED.value,
        CompositionEventType.EXPECTATION_UNAVAILABLE.value,
        CompositionEventType.EXPECTATION_MATCHED.value,
        CompositionEventType.EXPECTATION_MISMATCHED.value,
    }
    rows = []
    for event in trace:
        data = _event_dict(event)
        if data.get("type") not in expectation_types:
            continue
        payload = data.get("payload", {})
        rows.append(
            {
                "sample_index": data.get(
                    "external_sample_index", data.get("step_index")
                ),
                "event_type": data["type"],
                "module_id": data.get("module_id"),
                "source_module_id": payload.get("source_module_id"),
                "target_module_id": payload.get("target_module_id"),
                "high_level_category_id": payload.get("high_level_category_id"),
                "expected_category_ids": payload.get(
                    "expected_category_ids", []
                ),
                "current_low_level_category_id": payload.get(
                    "current_low_level_category_id"
                ),
                "matched": payload.get("matched"),
                "confidence": payload.get("confidence"),
            }
        )
    return rows


def build_cross_module_resonance_table(
    trace: Iterable[TraceEvent],
) -> List[Dict[str, Any]]:
    """Build one row per cross-module resonance or mismatch result."""
    result_types = {
        CompositionEventType.CROSS_MODULE_EXPECTATION_UNKNOWN.value,
        CompositionEventType.CROSS_MODULE_RESONANCE.value,
        CompositionEventType.CROSS_MODULE_MISMATCH.value,
    }
    rows = []
    for event in trace:
        data = _event_dict(event)
        if data.get("type") not in result_types:
            continue
        payload = data.get("payload", {})
        rows.append(
            {
                "sample_index": data.get(
                    "external_sample_index", data.get("step_index")
                ),
                "result": data["type"],
                "high_level_category_id": payload.get("high_level_category_id"),
                "expected_category_ids": payload.get(
                    "expected_category_ids", []
                ),
                "current_low_level_category_id": payload.get(
                    "current_low_level_category_id"
                ),
                "matched": payload.get("matched"),
            }
        )
    return rows


def summarize_bidirectional_step(
    trace: Iterable[TraceEvent],
    sample_index: int,
) -> Dict[str, Any]:
    """Summarize selections, expectation, and cross-module result."""
    base = summarize_graph_step(trace, sample_index)
    expectations = [
        row
        for row in build_expectation_table(base["events"])
        if row["event_type"]
        in {
            CompositionEventType.EXPECTATION_UNAVAILABLE.value,
            CompositionEventType.EXPECTATION_MATCHED.value,
            CompositionEventType.EXPECTATION_MISMATCHED.value,
        }
    ]
    results = build_cross_module_resonance_table(base["events"])
    return {
        **base,
        "expectation": expectations[-1] if expectations else None,
        "cross_module_result": results[-1] if results else None,
    }
