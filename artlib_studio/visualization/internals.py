"""Internals Inspector visualization."""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..core.events import EventType
from ..core.recorder import TraceRecorder
from ..core.capabilities import Capability


def show_internals_inspector(adapter, recorder: TraceRecorder, current_event_index: int):
    st.markdown("---")
    st.header("ART Internals Inspector")

    if current_event_index < 0:
        st.write("No event selected.")
        return

    events = recorder.events
    curr_ev = events[current_event_index]
    sample_idx = curr_ev.payload.get("sample_index", -1)

    if sample_idx == -1:
        st.write("Current event is not associated with a sample.")
        return

    # Filter events for this sample (all and so-far)
    sample_events_all = [e for e in events if e.payload.get("sample_index") == sample_idx]
    sample_events_so_far = [e for e in events[:current_event_index+1] if e.payload.get("sample_index") == sample_idx]

    # 1. Basic Info
    st.subheader("Current State")
    cols = st.columns(4)
    cols[0].metric("Sample Index", sample_idx)

    # Find INPUT_RECEIVED for this sample
    input_ev = next((e for e in sample_events_all if e.type == EventType.INPUT_RECEIVED), None)
    if input_ev:
        cols[1].metric("Vigilance (ρ)", f"{input_ev.payload.get('rho', 0.0):.3f}")
        cols[2].metric("Choice (α)", f"{input_ev.payload.get('alpha', 0.0):.3f}")
        cols[3].metric("Learning (β)", f"{input_ev.payload.get('beta', 0.0):.3f}")

    st.write(f"**Event Type:** {curr_ev.type.value}")

    st.write("**Input Activity:**")
    if input_ev:
        if adapter.supports(Capability.COMPLEMENT_CODING):
            tab1, tab2 = st.tabs(["Prepared (Complement Coded)", "Raw"])
            with tab1:
                st.code(str(input_ev.payload.get("prepared_input", [])))
            with tab2:
                st.code(str(input_ev.payload.get("input", [])))
        else:
            st.code(str(input_ev.payload.get("prepared_input", [])))

    # 2. Category Competition Table & Choice Score Visualization
    if adapter.supports(Capability.CHOICE_SCORES):
        st.subheader("Category Competition")

        # Get match and selected info for determining status
        eval_events = [e for e in sample_events_all if e.type == EventType.CATEGORY_EVALUATED]
        selected_evs_so_far = [e for e in sample_events_so_far if e.type == EventType.CATEGORY_SELECTED]
        last_selected_id = selected_evs_so_far[-1].payload.get("category_id") if selected_evs_so_far else None

        if eval_events:
            # We can use adapter's build_competition_table
            comp_data = adapter.build_competition_table(events[:current_event_index+1], sample_idx)
            if comp_data:
                st.table(pd.DataFrame(comp_data))

            # Choice Score Visualization
            st.write("**Choice Scores**")
            choice_scores = {e.payload.get("category_id"): e.payload.get("choice_score") for e in eval_events if e.payload.get("choice_score") is not None}
            if choice_scores:
                cids = sorted(choice_scores.keys())
                scores = [choice_scores[cid] for cid in cids]

                # Highlight winner
                colors = ['gray'] * len(cids)
                if last_selected_id is not None and last_selected_id in cids:
                    winner_idx = cids.index(last_selected_id)
                    colors[winner_idx] = 'red'

                fig_bar, ax_bar = plt.subplots(figsize=(6, 2))
                bars = ax_bar.bar([f"Cat {c}" for c in cids], scores, color=colors)
                ax_bar.set_ylabel("Choice Score")
                if last_selected_id is not None:
                    ax_bar.set_title(f"Winning Category: {last_selected_id} (red)")
                st.pyplot(fig_bar)

    # 3. Match vs Vigilance
    if adapter.supports(Capability.MATCH_SCORES):
        st.subheader("Match vs Vigilance")
        current_match_ev = next((e for e in reversed(sample_events_so_far) if e.type == EventType.MATCH_TEST), None)
        if current_match_ev:
            m_score = current_match_ev.payload.get("match_score")
            vig = current_match_ev.payload.get("vigilance")
            passed = current_match_ev.payload.get("passed")
            cid = current_match_ev.payload.get("category_id")

            st.write(f"Testing Category **{cid}**:")
            col1, col2, col3 = st.columns(3)
            col1.metric("Match Score", f"{m_score:.4f}")
            col2.metric("Vigilance", f"{vig:.4f}")
            col3.metric("Result", "PASSED" if passed else "FAILED (RESET)")

            # Simple progress bar style viz
            st.write("Match Score vs Vigilance Threshold:")
            st.progress(min(max(m_score, 0.0), 1.0))
            st.caption(f"Threshold at {vig:.4f}")
        else:
            st.write("No match test performed yet for the current step.")

    # 4. Search History
    if adapter.supports(Capability.RESET_SEARCH):
        st.subheader("Reset/Search History")
        history = []
        step_num = 1
        for e in sample_events_so_far:
            if e.type == EventType.MATCH_TEST:
                cid = e.payload.get("category_id")
                m_score = e.payload.get("match_score")
                vig = e.payload.get("vigilance")
                passed = e.payload.get("passed")
                if not passed:
                    history.append(f"{step_num}. Category {cid} selected. Match {m_score:.4f} < vigilance {vig:.4f}. Reset category {cid}.")
                    step_num += 1
            elif e.type == EventType.RESONANCE:
                cid = e.payload.get("category_id")
                m_score = e.payload.get("match_score")
                vig = e.payload.get("vigilance")
                history.append(f"{step_num}. Category {cid} selected. Match {m_score:.4f} >= vigilance {vig:.4f}. Resonance achieved.")
                step_num += 1
            elif e.type == EventType.CATEGORY_CREATED:
                history.append(f"{step_num}. No suitable category found. New category {e.payload.get('created_index')} created.")
                step_num += 1

        if history:
            for h in history:
                st.write(h)
        else:
            st.write("No history for current sample yet.")

    # 5. Learning / Weight Update View
    if adapter.supports(Capability.LEARNING_UPDATES):
        st.subheader("Learning update")
        learn_ev = next((e for e in sample_events_so_far if e.type == EventType.LEARNING), None)
        if learn_ev:
            cid = learn_ev.payload.get("category_id")
            w_before = learn_ev.payload.get("weights_before")
            w_after = learn_ev.payload.get("weights_after")

            st.write(f"Category **{cid}** weights update:")
            if adapter.supports(Capability.WEIGHT_BEFORE_AFTER) and w_before and w_after:
                df_w = pd.DataFrame({
                    "Dimension": range(len(w_before)),
                    "Before": w_before,
                    "After": w_after,
                    "Delta": np.array(w_after) - np.array(w_before)
                })
                st.table(df_w)

                if adapter.supports(Capability.COMPLEMENT_CODING) and len(w_before) % 2 == 0:
                    half = len(w_before) // 2
                    st.info(f"Note: Dimensions 0-{half-1} are original features. Dimensions {half}-{len(w_before)-1} are complement features (1-x).")
            else:
                st.write("Weights updated.")
        else:
            st.write("No learning event for current sample yet.")

    # 6. Conceptual Network View
    st.subheader("Conceptual ART Circuit")
    st.write("“This is a conceptual visualization of the ART computation, not a biological neuron simulation.”")

    f1_activity = input_ev.payload.get("prepared_input", []) if input_ev else []

    selected_evs_so_far = [e for e in sample_events_so_far if e.type == EventType.CATEGORY_SELECTED]
    last_selected_id = selected_evs_so_far[-1].payload.get("category_id") if selected_evs_so_far else None
    selected_cat = last_selected_id if last_selected_id is not None else "?"

    resonance_ev_so_far = next((e for e in sample_events_so_far if e.type == EventType.RESONANCE), None)

    if resonance_ev_so_far:
        status_text = "RESONANCE"
    elif any(r.type == EventType.RESET for r in sample_events_so_far):
        status_text = "RESET / SEARCHING"
    elif selected_evs_so_far:
        status_text = f"COMPARING CAT {last_selected_id}"
    else:
        status_text = "EVALUATING"

    circuit_text = f"""
    Input / F1 field: {f1_activity}
         ↓
    F2 category field (Competition)
         ↓
    Selected category: {selected_cat}
         ↓
    Top-down expectation (Weights)
         ↓
    Match comparison: {status_text}
    """
    st.code(circuit_text)

