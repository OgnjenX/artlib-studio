"""Config-driven graph composer for Streamlit."""
from __future__ import annotations

import json
from pathlib import Path
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
from ...composition.signals import InputSignal
from .composition_view import (
    CONFIG_DIR,
    render_composition_event_timeline,
    render_cross_module_resonance_table,
    render_edge_table,
    render_expectation_table,
    render_graph_overview,
    render_modulation_table,
    render_module_table,
    render_signal_flow_table,
)


COMPOSER_DATA = np.array(
    [[0.10, 0.12], [0.12, 0.14], [0.82, 0.85], [0.80, 0.83]],
    dtype=float,
)


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


def _set_composer_data(data: Dict[str, Any]) -> None:
    st.session_state.composer_data = data
    st.session_state.composer_revision = (
        st.session_state.get("composer_revision", 0) + 1
    )
    st.session_state.pop("composer_graph", None)


def _load_builtin(name: str) -> Dict[str, Any]:
    path = {
        "Two-level pipeline": CONFIG_DIR / "two_level_fuzzy_art.yaml",
        "Bidirectional expectation": CONFIG_DIR
        / "bidirectional_expectation.yaml",
        "Modulatory vigilance": CONFIG_DIR / "modulatory_vigilance.yaml",
    }[name]
    return graph_config_to_dict(load_graph_config(path))


def render_graph_composer() -> None:
    st.subheader("Graph Composer")
    st.caption(
        "This is a config-driven visual composer. It uses forms and a static "
        "preview rather than drag-and-drop."
    )
    if "composer_data" not in st.session_state:
        _set_composer_data(_load_builtin("Two-level pipeline"))

    built_in = st.selectbox(
        "Load built-in config",
        [
            "Two-level pipeline",
            "Bidirectional expectation",
            "Modulatory vigilance",
        ],
    )
    if st.button("Load selected config"):
        _set_composer_data(_load_builtin(built_in))
        st.rerun()

    pasted = st.text_area(
        "YAML / JSON configuration",
        value=yaml.safe_dump(st.session_state.composer_data, sort_keys=False),
        key=f"composer_yaml_{st.session_state.get('composer_revision', 0)}",
        height=300,
    )
    parse_cols = st.columns(2)
    if parse_cols[0].button("Validate configuration"):
        try:
            parsed = yaml.safe_load(pasted)
            config = graph_config_from_dict(parsed)
            st.session_state.composer_data = graph_config_to_dict(config)
            st.success("Configuration is valid.")
        except Exception as exc:
            st.error(str(exc))
    uploaded = parse_cols[1].file_uploader(
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
            _set_composer_data(graph_config_to_dict(graph_config_from_dict(parsed)))
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    data = st.session_state.composer_data
    with st.expander("Add module"):
        with st.form("composer_add_module"):
            module_id = st.text_input("Module id")
            module_type = st.selectbox("Module type", ["adapter", "context"])
            rho = st.number_input("rho / context value", 0.0, 1.0, 0.85)
            submitted = st.form_submit_button("Add module")
            if submitted:
                module = (
                    {
                        "id": module_id,
                        "type": "adapter",
                        "adapter": "fuzzy_art",
                        "params": {"rho": rho, "alpha": 0.001, "beta": 1.0},
                    }
                    if module_type == "adapter"
                    else {
                        "id": module_id,
                        "type": "context",
                        "rules": [
                            {
                                "name": "vigilance_context",
                                "active": True,
                                "target_param": "rho",
                                "mode": "set",
                                "value": rho,
                                "duration": "current_step",
                            }
                        ],
                    }
                )
                try:
                    _set_composer_data(add_module_to_config(data, module))
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    module_ids = [module["id"] for module in data.get("modules", [])]
    if module_ids:
        with st.expander("Remove module"):
            remove_id = st.selectbox("Module", module_ids, key="remove_module_id")
            if st.button("Remove module"):
                _set_composer_data(remove_module_from_config(data, remove_id))
                st.rerun()

    if len(module_ids) >= 2:
        with st.expander("Add edge"):
            with st.form("composer_add_edge"):
                source = st.selectbox("Source", module_ids)
                target = st.selectbox("Target", module_ids, index=1)
                edge_type = st.selectbox(
                    "Edge type",
                    [
                        "BOTTOM_UP",
                        "TOP_DOWN_EXPECTATION",
                        "MODULATORY",
                        "ASSOCIATIVE",
                        "RESET_PROPAGATION",
                    ],
                )
                transform_name = st.selectbox(
                    "Transform",
                    [
                        "none",
                        "selected_category_to_one_hot",
                        "selected_category_to_scalar_vector",
                        "selected_category_to_activation_vector",
                        "high_category_to_expectation",
                    ],
                )
                transform_value = st.number_input(
                    "vector_size / max_category_id",
                    min_value=1,
                    value=4,
                )
                submitted = st.form_submit_button("Add edge")
                if submitted:
                    if source == target:
                        st.error(
                            "Self-loops require advanced mode and are not "
                            "available in this composer."
                        )
                        st.stop()
                    edge = {
                        "source": source,
                        "target": target,
                        "type": edge_type,
                    }
                    if transform_name != "none":
                        params = {}
                        if transform_name == "selected_category_to_one_hot":
                            params["vector_size"] = int(transform_value)
                        elif transform_name == "selected_category_to_scalar_vector":
                            params["max_category_id"] = int(transform_value)
                        edge["transform"] = {
                            "name": transform_name,
                            "params": params,
                        }
                    try:
                        updated = add_edge_to_config(data, edge)
                        graph_config_from_dict(updated)
                        _set_composer_data(updated)
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

        if data.get("edges"):
            edge_labels = [
                f"{index}: {edge['source']} -> {edge['target']} ({edge['type']})"
                for index, edge in enumerate(data["edges"])
            ]
            remove_edge = st.selectbox(
                "Remove edge",
                edge_labels,
                key="remove_edge_index",
            )
            if st.button("Remove selected edge"):
                index = int(remove_edge.split(":", 1)[0])
                _set_composer_data(remove_edge_from_config(data, index))
                st.rerun()

    with st.expander("Association memory"):
        with st.form("composer_add_association"):
            high_category = st.number_input(
                "High-level category", min_value=0, value=0
            )
            low_category = st.number_input(
                "Low-level category", min_value=0, value=0
            )
            if st.form_submit_button("Add association"):
                updated = json.loads(json.dumps(data))
                updated.setdefault("associations", []).append(
                    {
                        "high_category_id": int(high_category),
                        "low_category_id": int(low_category),
                    }
                )
                _set_composer_data(updated)
                st.rerun()
        associations = data.get("associations", [])
        if associations:
            labels = [
                f"{index}: high {item['high_category_id']} -> "
                f"low {item['low_category_id']}"
                for index, item in enumerate(associations)
            ]
            selected = st.selectbox(
                "Remove association",
                labels,
                key="remove_association_index",
            )
            if st.button("Remove selected association"):
                updated = json.loads(json.dumps(data))
                updated["associations"].pop(int(selected.split(":", 1)[0]))
                _set_composer_data(updated)
                st.rerun()

    try:
        config = graph_config_from_dict(st.session_state.composer_data)
        preview_graph = build_graph_from_config(config)
        render_graph_overview(preview_graph)
        render_module_table(preview_graph.get_global_state())
        render_edge_table(preview_graph)
    except Exception as exc:
        st.error(f"Preview unavailable: {exc}")
        return

    if st.button("Run composed graph"):
        graph = build_graph_from_config(config)
        input_module = next(
            module.id for module in config.modules if module.type == "adapter"
        )
        for sample_index, sample in enumerate(COMPOSER_DATA):
            graph.step(
                [
                    InputSignal(
                        "environment",
                        target_module_id=input_module,
                        step_index=sample_index,
                        payload={"input": sample.tolist()},
                    )
                ]
            )
        st.session_state.composer_graph = graph

    yaml_text = config_dict_to_yaml(st.session_state.composer_data)
    json_text = config_dict_to_json(st.session_state.composer_data)
    download_cols = st.columns(2)
    download_cols[0].download_button(
        "Download YAML config",
        yaml_text,
        file_name=f"{config.name}.yaml",
    )
    download_cols[1].download_button(
        "Download JSON config",
        json_text,
        file_name=f"{config.name}.json",
    )

    graph = st.session_state.get("composer_graph")
    if graph is not None:
        trace = graph.get_event_log()
        render_graph_overview(graph)
        render_module_table(graph.get_global_state())
        render_signal_flow_table(trace)
        render_modulation_table(trace)
        render_expectation_table(trace)
        render_cross_module_resonance_table(trace)
        render_composition_event_timeline(trace)
        trace_json = json.dumps([event.to_dict() for event in trace], indent=2)
        st.download_button(
            "Download event trace JSON",
            trace_json,
            file_name=f"{config.name}_trace.json",
        )
