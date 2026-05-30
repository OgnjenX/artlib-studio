"""A minimal Streamlit-based ART Execution Explorer.

Run with: streamlit run artlib_studio/apps/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from typing import Tuple

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Make both `artlib_studio` and `AdaptiveResonanceLib` importable when running
# the file directly via `streamlit run artlib_studio/apps/streamlit_app.py`.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARL_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "AdaptiveResonanceLib"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if ARL_PATH not in sys.path:
    sys.path.append(ARL_PATH)

from sklearn.datasets import make_blobs

from artlib_studio.core.registry import list_adapters, get_adapter
from artlib_studio.core.events import EventType
from artlib_studio.core.recorder import TraceRecorder
from artlib_studio.core.capabilities import Capability
from artlib_studio.visualization.category_geometry import plot_state
from artlib_studio.visualization.internals import show_internals_inspector
from artlib_studio.visualization.process_tracker import render_process_visualization
from artlib_studio.visualization.match_vigilance import show_match_vigilance_panel
from artlib_studio.visualization.competition_table import show_competition_table
from artlib_studio.visualization.search_history import show_search_history
from artlib_studio.visualization.explanations import get_explanation

def make_dataset(n_samples=200, centers=3, seed=0) -> Tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=seed)
    # normalize to [0,1]
    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)
    Xn = (X - Xmin) / (Xmax - Xmin + 1e-9)
    return Xn, y


def main():
    st.set_page_config(layout="wide")
    st.title("ARTLib Studio — Step-By-Step ART Explorer (v0.4)")

    # Fetch adapters
    adapters = list_adapters()
    adapter_models = {a.name: a.model_key for a in adapters if a.supports(Capability.STREAMLIT_EXPLORER)}

    if not adapter_models:
        st.error("No compatible ART adapters found.")
        return

    if "recorder" not in st.session_state:
        st.session_state.recorder = None
        st.session_state.art = None
        st.session_state.instrumented = None
        st.session_state.X = None
        st.session_state.labels = None
        st.session_state.current_event_index = -1
        st.session_state.current_sample_filter = 0

    col_left, col_main, col_right = st.columns([1, 2, 1.5])

    with col_left:
        st.subheader("Data Settings")
        n = st.slider("n samples", 5, 200, 20)
        centers = st.slider("centers", 1, 6, 3)
        seed = st.number_input("seed", value=42)

        if st.button("Generate dataset"):
            X, y = make_dataset(n_samples=n, centers=centers, seed=seed)
            st.session_state.X = X
            st.session_state.labels = None
            st.session_state.recorder = None
            st.session_state.current_event_index = -1

        st.markdown("---")
        st.subheader("Model Settings")
        selected_model_name = st.selectbox("Select ART Model", list(adapter_models.keys()))
        selected_model_key = adapter_models[selected_model_name]
        adapter = get_adapter(selected_model_key)

        # Render dynamic parameters based on schema
        schema = adapter.param_schema()
        params = {}
        for p_key, p_info in schema.items():
            if p_info["type"] == "float":
                params[p_key] = st.slider(
                    p_info.get("label", p_key),
                    float(p_info.get("min", 0.0)),
                    float(p_info.get("max", 1.0)),
                    float(p_info.get("default", 0.5)),
                    step=float(p_info.get("step", 0.01))
                )

        if st.button(f"Run {selected_model_name}"):
            if st.session_state.X is None:
                st.warning("Generate dataset first")
            else:
                recorder = TraceRecorder()
                # Run through adapter
                instr = adapter.fit_with_trace(st.session_state.X, params, recorder=recorder)

                st.session_state.recorder = recorder
                st.session_state.art = instr.art
                st.session_state.instrumented = instr
                st.session_state.labels = getattr(instr.art, "labels_", None)
                st.session_state.current_event_index = 0 if len(recorder.events) > 0 else -1

        if st.button("Reset Explorer"):
            st.session_state.recorder = None
            st.session_state.art = None
            st.session_state.instrumented = None
            st.session_state.labels = None
            st.session_state.current_event_index = -1

        st.markdown("---")
        st.subheader("View Mode")
        view_mode = st.radio("Select View:", ["Timeline replay", "Final learned state"], index=0)
        show_future_faintly = st.checkbox("Show future samples faintly", value=False)
        
        st.markdown("---")
        if view_mode == "Timeline replay":
            st.info("ART learns incrementally. Timeline replay shows only samples and categories that exist up to the selected event. Future samples/categories are hidden or shown faintly.")
        else:
            st.warning("Final Learned State shows the final model state after all training is complete.")

    with col_main:
        st.subheader("Visualization")
        if st.session_state.recorder is None:
            st.info("Run model to see the visualization.")
            if st.session_state.X is not None:
                # Just show the data
                fig, ax = plt.subplots(figsize=(6,4))
                ax.scatter(st.session_state.X[:,0], st.session_state.X[:,1], s=20, color="gray")
                ax.set_title("Input Data")
                st.pyplot(fig)
        else:
            rec = st.session_state.recorder
            adapter_in_use = get_adapter(selected_model_key)
            if view_mode == "Final learned state":
                fig = plot_state(st.session_state.X, st.session_state.labels, st.session_state.art, adapter_in_use, rec, st.session_state.current_event_index)
            else:
                fig = render_process_visualization(
                    st.session_state.X, 
                    st.session_state.labels, 
                    adapter_in_use, 
                    st.session_state.art, 
                    rec, 
                    st.session_state.current_event_index,
                    show_future_faintly=show_future_faintly
                )
            st.pyplot(fig)
            
            st.caption("Category regions are approximate projections of learned Fuzzy ART expectations. They are not hard decision boundaries. Overlap can be normal. Actual category selection is determined by choice scores and vigilance match.")

            # Add Panels
            if view_mode == "Timeline replay" and st.session_state.current_event_index >= 0:
                current_event = rec.events[st.session_state.current_event_index]
                show_match_vigilance_panel(current_event)
                
                c_idx = current_event.payload.get("sample_index", -1) if hasattr(current_event, "payload") else -1
                show_competition_table(rec, c_idx)
                
            # Add Internals Inspector
            if adapter_in_use.supports(Capability.TRACE_EXECUTION):
                show_internals_inspector(adapter_in_use, rec, st.session_state.current_event_index)

    with col_right:
        st.subheader("Event Trace")
        if st.session_state.recorder is None:
            st.write("No trace available.")
        else:
            rec = st.session_state.recorder
            adapter_in_use = get_adapter(selected_model_key)

            curr_ev = rec.events[st.session_state.current_event_index] if st.session_state.current_event_index >= 0 else None
            c_idx = curr_ev.payload.get("sample_index", -1) if curr_ev else -1
            
            # Progress tracking
            total_samples = len(st.session_state.X) if st.session_state.X is not None else 0
            if view_mode == "Timeline replay" and c_idx >= 0:
                from artlib_studio.visualization.timeline_state import build_timeline_state
                t_state = build_timeline_state(st.session_state.X, rec, st.session_state.current_event_index)
                proc_samps = len(t_state["processed_sample_indices"]) if t_state else 0
                st.write(f"**Processed samples:** {proc_samps} / {total_samples}")
                st.write(f"**Current event:** {st.session_state.current_event_index + 1} / {len(rec.events)}")
                st.write(f"**Current sample:** {c_idx}")

            # Global Step Controls
            st.write("**Global Step Controls**")
            c1, c2 = st.columns([1, 1])
            if c1.button("<< Prev Event"):
                if st.session_state.current_event_index > 0:
                    st.session_state.current_event_index -= 1
            if c2.button("Next Event >>"):
                if st.session_state.current_event_index < len(rec.events) - 1:
                    st.session_state.current_event_index += 1

            c3, c4 = st.columns([1, 1])
            if c3.button("<< Prev Sample"):
                if c_idx > 0:
                    for i in range(st.session_state.current_event_index - 1, -1, -1):
                        if rec.events[i].payload.get("sample_index", -1) < c_idx:
                            for j in range(i, -1, -1):
                                if j == 0 or rec.events[j-1].payload.get("sample_index", -1) < rec.events[i].payload.get("sample_index", -1):
                                    st.session_state.current_event_index = j
                                    break
                            break
            if c4.button("Next Sample >>"):
                if c_idx >= 0:
                    for i in range(st.session_state.current_event_index + 1, len(rec.events)):
                        if rec.events[i].payload.get("sample_index", -1) > c_idx:
                            st.session_state.current_event_index = i
                            break
                        
            if st.button("Jump to first event of sample"):
                if c_idx >= 0:
                    for i in range(st.session_state.current_event_index, -1, -1):
                        if i == 0 or rec.events[i-1].payload.get("sample_index", -1) < c_idx:
                            st.session_state.current_event_index = i
                            break
                            
            # Update to fetch re-evaluated index if changed by buttons
            curr_ev = rec.events[st.session_state.current_event_index] if st.session_state.current_event_index >= 0 else None
            c_idx = curr_ev.payload.get("sample_index", -1) if curr_ev else -1

            if curr_ev:
                st.markdown(f"**Current Event ({st.session_state.current_event_index + 1}/{len(rec.events)})**")
                explanation = get_explanation(curr_ev)
                st.info(explanation)
                with st.expander("Event Payload"):
                    st.json(curr_ev.payload)

            st.markdown("---")
            if view_mode == "Timeline replay":
                show_search_history(rec, c_idx)
            else:
                # Group events by sample
                sample_ids = sorted(list(set([e.payload.get("sample_index", -1) for e in rec.events if e.payload.get("sample_index", -1) != -1])))
                if sample_ids:
                    selected_sample = st.selectbox("Inspect specific sample trace:", sample_ids, index=0)
                    sample_events = [e for e in rec.events if e.payload.get("sample_index", -1) == selected_sample]
    
                    st.write(f"**Trace for Sample {selected_sample}:**")
                    for e in sample_events:
                        color = "blue" if e.type == EventType.RESONANCE else ("green" if e.type == EventType.CATEGORY_CREATED else "black")
                        explanation = get_explanation(e)
                        st.markdown(f"- :{color}[{e.type.value}]: {explanation}")

if __name__ == "__main__":
    main()
