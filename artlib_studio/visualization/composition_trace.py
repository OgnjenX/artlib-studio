"""Non-UI views over composition event traces."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Union

from ..composition.events import CompositionEvent, CompositionEventType

TraceEvent = Union[CompositionEvent, Dict[str, Any]]


def _event_dict(event: TraceEvent) -> Dict[str, Any]:
    if isinstance(event, CompositionEvent):
        return event.to_dict()
    return dict(event)


def build_module_state_table(graph_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten graph module state for tables and exports."""
    rows = []
    for module_id, state in graph_state.get("modules", {}).items():
        params = state.get("params", {})
        rows.append(
            {
                "module_id": module_id,
                "type": state.get("type", "adapter"),
                "adapter": state.get("adapter"),
                "selected_category": state.get("selected_category"),
                "category_count": state.get("category_count"),
                "sample_count": state.get("sample_count"),
                "pending_signals": state.get("pending_signals", 0),
                "rho": params.get("rho"),
                "active_rules": state.get("active_rules"),
            }
        )
    return rows


def build_edge_table(graph: Any) -> List[Dict[str, Any]]:
    """Flatten graph edges for tables and static previews."""
    return [
        {
            "source": edge.source_module_id,
            "target": edge.target_module_id,
            "edge_type": edge.edge_type.value,
            "transform": edge.transform_name,
        }
        for edge in graph.edges
    ]


def build_graph_overview(graph: Any) -> Dict[str, Any]:
    """Return a serializable overview of modules, edges, and graph state."""
    state = graph.get_global_state()
    return {
        "step_index": state["step_index"],
        "edge_count": state["edge_count"],
        "event_count": state["event_count"],
        "associations": state.get("associations", {}),
        "modules": build_module_state_table(state),
        "edges": build_edge_table(graph),
    }


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


def build_modulation_table(trace: Iterable[TraceEvent]) -> List[Dict[str, Any]]:
    """Build rows showing modulation delivery, application, and restoration."""
    modulation_types = {
        CompositionEventType.MODULATION_SENT.value,
        CompositionEventType.MODULATION_RECEIVED.value,
        CompositionEventType.MODULE_PARAMETER_MODULATED.value,
        CompositionEventType.MODULE_PARAMETER_RESTORED.value,
    }
    rows = []
    for event in trace:
        data = _event_dict(event)
        if data.get("type") not in modulation_types:
            continue
        payload = data.get("payload", {})
        event_type = data["type"]
        rows.append(
            {
                "sample_index": data.get(
                    "external_sample_index", data.get("step_index")
                ),
                "event_type": event_type,
                "module_id": data.get("module_id"),
                "source_module_id": payload.get("source_module_id")
                or (
                    data.get("module_id")
                    if event_type
                    == CompositionEventType.MODULATION_SENT.value
                    else None
                ),
                "target_module_id": payload.get("target_module_id")
                or (
                    data.get("module_id")
                    if event_type
                    == CompositionEventType.MODULE_PARAMETER_RESTORED.value
                    else None
                ),
                "target_param": payload.get("target_param"),
                "mode": payload.get("mode"),
                "before": payload.get("before"),
                "after": payload.get("after"),
                "duration": payload.get("duration"),
                "restored_value": payload.get("restored_value"),
                "explanation": payload.get("explanation"),
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
