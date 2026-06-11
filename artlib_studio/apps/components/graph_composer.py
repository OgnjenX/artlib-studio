"""Interactive, config-driven graph composer for Streamlit."""
from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
import streamlit as st
import yaml

from ...composition.builder import build_graph_from_config
from ...composition.config import (
    graph_config_from_dict,
    graph_config_to_dict,
    load_graph_config,
)
from ...composition.events import CompositionEventType
from ...composition.signals import InputSignal
from ...graph_canvas_component import art_graph_canvas
from .composition_view import (
    CONFIG_DIR,
    render_composition_event_timeline,
    render_cross_module_resonance_table,
    render_expectation_table,
    render_modulation_table,
    render_module_table,
    render_signal_flow_table,
)


COMPOSER_DATA = np.array(
    [[0.10, 0.12], [0.12, 0.14], [0.82, 0.85], [0.80, 0.83]],
    dtype=float,
)
CANVAS_KEY = "art_composition_canvas"


def empty_graph_config_dict(name: str = "composed_graph") -> Dict[str, Any]:
    return {
        "name": name,
        "description": "Config-driven ART composition graph",
        "modules": [],
        "edges": [],
        "associations": [],
        "max_settling_steps": 5,
    }


def add_module_to_config(
    data: Dict[str, Any],
    module: Dict[str, Any],
) -> Dict[str, Any]:
    updated = json.loads(json.dumps(data))
    if any(item["id"] == module["id"] for item in updated["modules"]):
        raise ValueError(f"Module id {module['id']!r} already exists")
    updated["modules"].append(module)
    return updated


def remove_module_from_config(
    data: Dict[str, Any],
    module_id: str,
) -> Dict[str, Any]:
    updated = json.loads(json.dumps(data))
    updated["modules"] = [
        module for module in updated["modules"] if module["id"] != module_id
    ]
    updated["edges"] = [
        edge
        for edge in updated["edges"]
        if edge["source"] != module_id and edge["target"] != module_id
    ]
    return updated


def add_edge_to_config(
    data: Dict[str, Any],
    edge: Dict[str, Any],
) -> Dict[str, Any]:
    updated = json.loads(json.dumps(data))
    updated["edges"].append(edge)
    return updated


def remove_edge_from_config(
    data: Dict[str, Any],
    edge_index: int,
) -> Dict[str, Any]:
    updated = json.loads(json.dumps(data))
    if not 0 <= edge_index < len(updated["edges"]):
        raise ValueError("Edge index is out of range")
    updated["edges"].pop(edge_index)
    return updated


def config_dict_to_yaml(data: Dict[str, Any]) -> str:
    graph_config_from_dict(data)
    return yaml.safe_dump(data, sort_keys=False)


def config_dict_to_json(data: Dict[str, Any]) -> str:
    graph_config_from_dict(data)
    return json.dumps(data, indent=2)


def _load_builtin(name: str) -> Dict[str, Any]:
    path = {
        "Two-level pipeline": CONFIG_DIR / "two_level_fuzzy_art.yaml",
        "Bidirectional expectation": CONFIG_DIR
        / "bidirectional_expectation.yaml",
        "Modulatory vigilance": CONFIG_DIR / "modulatory_vigilance.yaml",
    }[name]
    return graph_config_to_dict(load_graph_config(path))


def _set_composer_data(data: Dict[str, Any]) -> None:
    config = graph_config_from_dict(data)
    st.session_state.composer_data = graph_config_to_dict(config)
    st.session_state.composer_revision = (
        st.session_state.get("composer_revision", 0) + 1
    )
    st.session_state.composer_canvas_revision = (
        st.session_state.get("composer_canvas_revision", 0) + 1
    )
    _reset_runtime()


def _reset_runtime() -> None:
    st.session_state.pop("composer_graph", None)
    st.session_state.composer_sample_index = -1


def _canvas_graph_changed() -> None:
    component_state = st.session_state.get(CANVAS_KEY)
    candidate = getattr(component_state, "graph", None)
    if candidate is None and isinstance(component_state, dict):
        candidate = component_state.get("graph")
    if candidate is None:
        return
    try:
        config = graph_config_from_dict(candidate)
    except (KeyError, TypeError, ValueError) as exc:
        st.session_state.composer_canvas_error = str(exc)
        st.session_state.composer_canvas_revision = (
            st.session_state.get("composer_canvas_revision", 0) + 1
        )
        return
    st.session_state.composer_data = graph_config_to_dict(config)
    st.session_state.composer_revision = (
        st.session_state.get("composer_revision", 0) + 1
    )
    st.session_state.pop("composer_canvas_error", None)
    _reset_runtime()


def _ensure_runtime(config) -> Any:
    graph = st.session_state.get("composer_graph")
    if graph is None:
        graph = build_graph_from_config(config)
        st.session_state.composer_graph = graph
    return graph


def _input_module_id(config) -> str:
    try:
        return next(
            module.id for module in config.modules if module.type == "adapter"
        )
    except StopIteration as exc:
        raise ValueError("The graph needs at least one ART adapter module") from exc


def _step_runtime(config) -> None:
    next_index = st.session_state.get("composer_sample_index", -1) + 1
    if next_index >= len(COMPOSER_DATA):
        raise ValueError(
            f"Demo dataset complete ({len(COMPOSER_DATA)} samples). "
            "Reset runtime to replay it."
        )
    graph = _ensure_runtime(config)
    graph.step(
        [
            InputSignal(
                "environment",
                target_module_id=_input_module_id(config),
                step_index=next_index,
                payload={"input": COMPOSER_DATA[next_index].tolist()},
            )
        ]
    )
    st.session_state.composer_sample_index = next_index


def _runtime_payload(graph: Any) -> Dict[str, Any]:
    if graph is None:
        return {
            "step_index": 0,
            "modules": {},
            "active_modules": [],
            "active_edges": [],
            "last_event": None,
        }
    state = graph.get_global_state()
    current_step = max(0, state["step_index"] - 1)
    step_events = [
        event
        for event in graph.get_event_log()
        if event.step_index == current_step
    ]
    active_module_types = {
        CompositionEventType.MODULE_RECEIVED_INPUT,
        CompositionEventType.MODULE_SELECTED_CATEGORY,
        CompositionEventType.MODULE_RESONATED,
        CompositionEventType.MODULE_RESET,
        CompositionEventType.MODULE_LEARNED,
        CompositionEventType.MODULE_PARAMETER_MODULATED,
    }
    active_modules = sorted(
        {
            event.module_id
            for event in step_events
            if event.module_id is not None and event.type in active_module_types
        }
    )
    active_edges = sorted(
        {
            f"{event.module_id}:{event.payload['target_module_id']}"
            for event in step_events
            if event.type == CompositionEventType.SIGNAL_SENT
            and event.module_id is not None
            and event.payload.get("target_module_id") is not None
        }
    )
    return {
        "step_index": state["step_index"],
        "modules": state["modules"],
        "active_modules": active_modules,
        "active_edges": active_edges,
        "last_event": step_events[-1].type.value if step_events else None,
    }


def _render_config_io() -> None:
    with st.expander("Config import / export"):
        built_in = st.selectbox(
            "Built-in graph",
            [
                "Two-level pipeline",
                "Bidirectional expectation",
                "Modulatory vigilance",
            ],
        )
        if st.button("Load built-in graph"):
            _set_composer_data(_load_builtin(built_in))
            st.rerun()

        pasted = st.text_area(
            "YAML / JSON configuration",
            value=yaml.safe_dump(
                st.session_state.composer_data,
                sort_keys=False,
            ),
            key=f"composer_yaml_{st.session_state.composer_revision}",
            height=280,
        )
        cols = st.columns(2)
        if cols[0].button("Apply configuration"):
            try:
                parsed = yaml.safe_load(pasted)
                _set_composer_data(graph_config_to_dict(
                    graph_config_from_dict(parsed)
                ))
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        uploaded = cols[1].file_uploader(
            "Upload YAML or JSON",
            type=["yaml", "yml", "json"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            try:
                text = uploaded.getvalue().decode("utf-8")
                parsed = (
                    json.loads(text)
                    if uploaded.name.endswith(".json")
                    else yaml.safe_load(text)
                )
                _set_composer_data(graph_config_to_dict(
                    graph_config_from_dict(parsed)
                ))
                st.rerun()
            except Exception as exc:
                st.error(str(exc))


def render_graph_composer() -> None:
    st.subheader("Visual Graph Composer")
    st.caption(
        "Build ART networks directly on the canvas. Drag modules, connect "
        "ports, edit parameters, then step the executable graph."
    )
    if "composer_data" not in st.session_state:
        st.session_state.composer_revision = 0
        st.session_state.composer_canvas_revision = 0
        _set_composer_data(_load_builtin("Two-level pipeline"))

    _render_config_io()

    try:
        config = graph_config_from_dict(st.session_state.composer_data)
    except Exception as exc:
        st.error(f"Graph configuration is invalid: {exc}")
        return

    sample_index = st.session_state.get("composer_sample_index", -1)
    dataset_complete = sample_index + 1 >= len(COMPOSER_DATA)
    controls = st.columns([1, 1.2, 1, 2])
    if controls[0].button(
        "Step",
        type="primary",
        disabled=dataset_complete,
        help=(
            "Reset runtime to replay the demo dataset."
            if dataset_complete
            else "Process the next input sample."
        ),
    ):
        try:
            _step_runtime(config)
        except Exception as exc:
            st.session_state.composer_run_error = str(exc)
    if controls[1].button(
        "Run remaining",
        disabled=dataset_complete,
        help=(
            "All demo samples have been processed."
            if dataset_complete
            else "Process every remaining input sample."
        ),
    ):
        try:
            while (
                st.session_state.get("composer_sample_index", -1) + 1
                < len(COMPOSER_DATA)
            ):
                _step_runtime(config)
        except Exception as exc:
            st.session_state.composer_run_error = str(exc)
    if controls[2].button("Reset runtime"):
        _reset_runtime()
        st.session_state.pop("composer_run_error", None)
    if sample_index < 0:
        controls[3].caption(
            f"Ready: 0/{len(COMPOSER_DATA)} demo samples processed"
        )
    elif dataset_complete:
        controls[3].caption(
            f"Dataset complete: {len(COMPOSER_DATA)}/"
            f"{len(COMPOSER_DATA)} samples processed. Reset to replay."
        )
    else:
        controls[3].caption(
            f"Sample {sample_index + 1}/{len(COMPOSER_DATA)}: "
            f"{COMPOSER_DATA[sample_index].tolist()}"
        )

    canvas_error = st.session_state.get("composer_canvas_error")
    run_error = st.session_state.get("composer_run_error")
    if canvas_error:
        st.error(f"Canvas edit rejected: {canvas_error}")
    if run_error:
        st.error(f"Graph execution failed: {run_error}")

    graph = st.session_state.get("composer_graph")
    art_graph_canvas(
        st.session_state.composer_data,
        revision=st.session_state.composer_canvas_revision,
        runtime=_runtime_payload(graph),
        key=CANVAS_KEY,
        on_graph_change=_canvas_graph_changed,
    )

    try:
        config = graph_config_from_dict(st.session_state.composer_data)
        yaml_text = config_dict_to_yaml(st.session_state.composer_data)
        json_text = config_dict_to_json(st.session_state.composer_data)
    except Exception as exc:
        st.error(f"Canvas graph is invalid: {exc}")
        return

    download_cols = st.columns(3)
    download_cols[0].download_button(
        "Download YAML",
        yaml_text,
        file_name=f"{config.name}.yaml",
    )
    download_cols[1].download_button(
        "Download JSON",
        json_text,
        file_name=f"{config.name}.json",
    )

    graph = st.session_state.get("composer_graph")
    if graph is None:
        st.info(
            "Connect valid modules and press Step to see category selection "
            "and signal activity on the canvas."
        )
        return

    trace = graph.get_event_log()
    trace_json = json.dumps([event.to_dict() for event in trace], indent=2)
    download_cols[2].download_button(
        "Download trace",
        trace_json,
        file_name=f"{config.name}_trace.json",
    )

    with st.expander("Runtime details", expanded=False):
        render_module_table(graph.get_global_state())
        render_signal_flow_table(trace)
        render_modulation_table(trace)
        render_expectation_table(trace)
        render_cross_module_resonance_table(trace)
        render_composition_event_timeline(trace)
