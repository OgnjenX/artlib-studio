"""Streamlit rendering for composition graphs and traces."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import streamlit as st

from ...composition.builder import build_graph_from_config
from ...composition.config import load_graph_config
from ...composition.signals import InputSignal
from ...visualization.composition_trace import (
    build_cross_module_resonance_table,
    build_edge_table,
    build_expectation_table,
    build_graph_overview,
    build_modulation_table,
    build_module_state_table,
    build_module_timeline,
    build_signal_flow_table as make_signal_flow_table,
    summarize_bidirectional_step,
)


CONFIG_DIR = Path(__file__).resolve().parents[3] / "examples" / "configs"
DEMO_DATA = np.array(
    [[0.10, 0.12], [0.12, 0.14], [0.82, 0.85], [0.80, 0.83]],
    dtype=float,
)


def render_composition_experiment_selector() -> str:
    return st.selectbox(
        "Composition experiment",
        [
            "Two-Level Feed-Forward Pipeline",
            "Bidirectional Expectation Prototype",
            "Modulatory Vigilance Demo",
            "Graph Composer",
        ],
    )


def render_graph_overview(graph: Any) -> None:
    overview = build_graph_overview(graph)
    st.subheader("Graph Overview")
    diagram = []
    input_module = next(
        (
            module["module_id"]
            for module in overview["modules"]
            if module.get("adapter") is not None
        ),
        None,
    )
    if input_module is not None:
        diagram.append(f"[Environment] --> [{input_module}]")
    for edge in overview["edges"]:
        diagram.append(
            f"[{edge['source']}] -- {edge['edge_type']} / "
            f"{edge['transform'] or 'identity'} --> "
            f"[{edge['target']}]"
        )
    st.code("\n".join(diagram), language="text")
    cols = st.columns(4)
    cols[0].metric("Step", overview["step_index"])
    cols[1].metric("Modules", len(overview["modules"]))
    cols[2].metric("Edges", overview["edge_count"])
    cols[3].metric("Events", overview["event_count"])
    if overview["associations"]:
        st.caption(f"Association memory: {overview['associations']}")


def render_module_table(graph_state: dict) -> None:
    st.subheader("Modules")
    st.dataframe(
        pd.DataFrame(build_module_state_table(graph_state)),
        width="stretch",
        hide_index=True,
    )


def render_edge_table(graph: Any) -> None:
    st.subheader("Edges")
    st.dataframe(
        pd.DataFrame(build_edge_table(graph)),
        width="stretch",
        hide_index=True,
    )


def render_signal_flow_table(trace: Iterable[Any]) -> None:
    st.subheader("Signal Flow")
    st.dataframe(
        pd.DataFrame(make_signal_flow_table(trace)),
        width="stretch",
        hide_index=True,
    )


def render_composition_event_timeline(trace: Iterable[Any]) -> None:
    st.subheader("Graph Event Timeline")
    st.dataframe(
        pd.DataFrame(build_module_timeline(trace)),
        width="stretch",
        hide_index=True,
    )


def render_expectation_table(trace: Iterable[Any]) -> None:
    rows = build_expectation_table(trace)
    if rows:
        st.subheader("Top-Down Expectations")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_modulation_table(trace: Iterable[Any]) -> None:
    rows = build_modulation_table(trace)
    if rows:
        st.subheader("Parameter Modulation")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_cross_module_resonance_table(trace: Iterable[Any]) -> None:
    rows = build_cross_module_resonance_table(trace)
    if rows:
        st.subheader("Cross-Module Result")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_bidirectional_summary(trace: Iterable[Any], sample_index: int) -> None:
    summary = summarize_bidirectional_step(trace, sample_index)
    result = summary.get("cross_module_result")
    if result is None:
        return
    if result["matched"] is True:
        st.success("Cross-module resonance: the expectation matched.")
    elif result["matched"] is False:
        st.error("Cross-module mismatch: the expectation was violated.")
    else:
        st.info("Expectation unknown: no prior association is available.")


def _config_for_experiment(experiment: str) -> Optional[Path]:
    return {
        "Two-Level Feed-Forward Pipeline": CONFIG_DIR
        / "two_level_fuzzy_art.yaml",
        "Bidirectional Expectation Prototype": CONFIG_DIR
        / "bidirectional_expectation.yaml",
        "Modulatory Vigilance Demo": CONFIG_DIR / "modulatory_vigilance.yaml",
    }.get(experiment)


def render_composition_experiment(experiment: str) -> None:
    config_path = _config_for_experiment(experiment)
    if config_path is None:
        from .graph_composer import render_graph_composer

        render_graph_composer()
        return

    key = experiment.lower().replace(" ", "_")
    graph_key = f"composition_graph_{key}"
    sample_key = f"composition_sample_{key}"
    if graph_key not in st.session_state:
        st.session_state[graph_key] = build_graph_from_config(
            load_graph_config(config_path)
        )
        st.session_state[sample_key] = -1

    graph = st.session_state[graph_key]
    st.caption(load_graph_config(config_path).description)
    if experiment == "Two-Level Feed-Forward Pipeline":
        st.info(
            "The high-level ART receives a one-hot encoding of the low-level "
            "selected category. This is an engineering simplification and not "
            "a full neural ART code."
        )
    elif experiment == "Bidirectional Expectation Prototype":
        st.info(
            "AssociationMemory supplies a simplified top-down expectation. "
            "Unknown, matched, and violated expectations are distinguished."
        )
    else:
        st.info(
            "The context node explicitly raises vigilance for each graph step. "
            "This is an educational control mechanism, not neuromodulation."
        )

    controls = st.columns(3)
    if controls[0].button("Process next sample", key=f"next_{key}"):
        next_index = st.session_state[sample_key] + 1
        if next_index < len(DEMO_DATA):
            target = next(
                module_id
                for module_id, module in graph.modules.items()
                if getattr(module, "adapter", None) is not None
            )
            graph.step(
                [
                    InputSignal(
                        "environment",
                        target_module_id=target,
                        step_index=next_index,
                        payload={"input": DEMO_DATA[next_index].tolist()},
                    )
                ]
            )
            st.session_state[sample_key] = next_index
    if controls[1].button("Run all", key=f"all_{key}"):
        while st.session_state[sample_key] + 1 < len(DEMO_DATA):
            next_index = st.session_state[sample_key] + 1
            target = next(
                module_id
                for module_id, module in graph.modules.items()
                if getattr(module, "adapter", None) is not None
            )
            graph.step(
                [
                    InputSignal(
                        "environment",
                        target_module_id=target,
                        step_index=next_index,
                        payload={"input": DEMO_DATA[next_index].tolist()},
                    )
                ]
            )
            st.session_state[sample_key] = next_index
    if controls[2].button("Reset", key=f"reset_{key}"):
        st.session_state[graph_key] = build_graph_from_config(
            load_graph_config(config_path)
        )
        st.session_state[sample_key] = -1
        st.rerun()

    render_graph_overview(graph)
    render_module_table(graph.get_global_state())
    render_edge_table(graph)
    trace = graph.get_event_log()
    if trace:
        current = st.session_state[sample_key]
        st.write(f"Current sample: {current}")
        render_bidirectional_summary(trace, current)
        render_signal_flow_table(trace)
        render_modulation_table(trace)
        render_expectation_table(trace)
        render_cross_module_resonance_table(trace)
        render_composition_event_timeline(trace)
