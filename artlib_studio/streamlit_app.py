"""A minimal Streamlit-based ART Execution Explorer.

Run with: streamlit run artlib_studio/streamlit_app.py
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
# the file directly via `streamlit run artlib_studio/streamlit_app.py`.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARL_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "AdaptiveResonanceLib"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if ARL_PATH not in sys.path:
    sys.path.append(ARL_PATH)

from artlib.elementary.FuzzyART import FuzzyART
from sklearn.datasets import make_blobs

from artlib_studio.instrumented_art import InstrumentedART
from artlib_studio.events import TraceRecorder, EventType


def make_dataset(n_samples=200, centers=3, seed=0) -> Tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=seed)
    # normalize to [0,1]
    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)
    Xn = (X - Xmin) / (Xmax - Xmin + 1e-9)
    return Xn, y


def plot_state(X, labels, art, recorder: TraceRecorder, current_event_index: int):
    fig, ax_points = plt.subplots(figsize=(6, 4))

    # Identify current sample if any
    events = recorder.events
    current_sample_idx = -1
    if 0 <= current_event_index < len(events):
        current_sample_idx = events[current_event_index].payload.get("sample_index", -1)

    # Panel: Data points and categories
    if labels is None:
        ax_points.scatter(X[:, 0], X[:, 1], s=20, color="gray")
    else:
        # color by label
        for k in np.unique(labels):
            mask = labels == k
            ax_points.scatter(X[mask, 0], X[mask, 1], s=20, label=f"c{k}")

    # Highlight current sample
    if current_sample_idx >= 0:
        ax_points.scatter(X[current_sample_idx, 0], X[current_sample_idx, 1],
                          s=150, facecolors='none', edgecolors='red', linewidths=2, label="Current")

    # Plot category bounding boxes (using final state for now as approximation)
    try:
        if hasattr(art, "get_bounding_boxes") and getattr(art, "n_clusters", 0) > 0:
            bboxes = art.get_bounding_boxes(n=2)
            for idx, bbox in enumerate(bboxes):
                ref, widths = bbox
                rect = plt.Rectangle((ref[0], ref[1]), widths[0], widths[1], fill=False, edgecolor="C%d" % (idx % 10), linewidth=2)
                ax_points.add_patch(rect)
                ax_points.text(ref[0], ref[1], f"Cat {idx}", color="C%d" % (idx % 10), fontsize=9)
    except Exception:
        pass

    ax_points.set_title("Data and Approximate Category Expectations")
    ax_points.set_xlim(-0.05, 1.05)
    ax_points.set_ylim(-0.05, 1.05)
    ax_points.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    return fig


def show_internals_inspector(recorder: TraceRecorder, current_event_index: int):
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
        tab1, tab2 = st.tabs(["Prepared (Complement Coded)", "Raw"])
        with tab1:
            st.code(str(input_ev.payload.get("prepared_input", [])))
        with tab2:
            st.code(str(input_ev.payload.get("input", [])))

    # 2. Category Competition Table & Choice Score Visualization
    st.subheader("Category Competition")

    # Use so_far for status, all for scores
    eval_events = [e for e in sample_events_all if e.type == EventType.CATEGORY_EVALUATED]
    match_events_so_far = [e for e in sample_events_so_far if e.type == EventType.MATCH_TEST]
    reset_events_so_far = [e for e in sample_events_so_far if e.type == EventType.RESET]
    resonance_ev_so_far = next((e for e in sample_events_so_far if e.type == EventType.RESONANCE), None)
    created_ev_so_far = next((e for e in sample_events_so_far if e.type == EventType.CATEGORY_CREATED), None)
    selected_evs_so_far = [e for e in sample_events_so_far if e.type == EventType.CATEGORY_SELECTED]
    last_selected_id = selected_evs_so_far[-1].payload.get("category_id") if selected_evs_so_far else None

    if eval_events:
        comp_data = []
        choice_scores = {}
        for ev in eval_events:
            cid = ev.payload.get("category_id")
            score = ev.payload.get("choice_score")
            choice_scores[cid] = score

            # Find match result if it was tested so far
            m_ev = next((e for e in match_events_so_far if e.payload.get("category_id") == cid), None)
            m_score = m_ev.payload.get("match_score") if m_ev else None
            vig = m_ev.payload.get("vigilance") if m_ev else None

            status = "not evaluated"
            if m_ev:
                if any(r.payload.get("category_id") == cid for r in reset_events_so_far):
                    status = "reset"
                elif resonance_ev_so_far and resonance_ev_so_far.payload.get("category_id") == cid:
                    status = "resonance"
                else:
                    status = "selected"
            elif last_selected_id is not None and cid == last_selected_id:
                status = "selected"
            elif any(e.type == EventType.CATEGORY_EVALUATED and e.payload.get("category_id") == cid for e in sample_events_so_far):
                status = "not selected"
            else:
                status = "not evaluated"

            comp_data.append({
                "Category": cid,
                "Choice score": f"{score:.4f}" if score is not None else "NaN",
                "Match score": f"{m_score:.4f}" if m_score is not None else "—",
                "Vigilance": f"{vig:.4f}" if vig is not None else "—",
                "Result": status
            })

        if created_ev_so_far:
            comp_data.append({
                "Category": created_ev_so_far.payload.get("created_index"),
                "Choice score": "—",
                "Match score": "—",
                "Vigilance": "—",
                "Result": "new category"
            })

        st.table(pd.DataFrame(comp_data))

        # Choice Score Visualization
        st.write("**Choice Scores**")
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
    st.subheader("Learning update")
    learn_ev = next((e for e in sample_events_so_far if e.type == EventType.LEARNING), None)
    if learn_ev:
        cid = learn_ev.payload.get("category_id")
        w_before = learn_ev.payload.get("weights_before")
        w_after = learn_ev.payload.get("weights_after")

        st.write(f"Category **{cid}** weights update:")
        if w_before and w_after:
            df_w = pd.DataFrame({
                "Dimension": range(len(w_before)),
                "Before": w_before,
                "After": w_after,
                "Delta": np.array(w_after) - np.array(w_before)
            })
            st.table(df_w)

            # Explain dimensions for complement coding
            if len(w_before) % 2 == 0:
                half = len(w_before) // 2
                st.info(f"Note: Dimensions 0-{half-1} are original features. Dimensions {half}-{len(w_before)-1} are complement features (1-x).")
        else:
            st.write("Weight data not available for this event.")
    else:
        st.write("No learning event for current sample yet.")

    # 6. Conceptual F1/F2 network view
    st.subheader("Conceptual ART Circuit")
    st.write("“This is a conceptual visualization of Fuzzy ART computation, not a biological neuron simulation.”")

    # Simple ASCII/Text diagram
    f1_activity = input_ev.payload.get("prepared_input", []) if input_ev else []
    selected_cat = last_selected_id if last_selected_id is not None else "?"

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


def main():
    st.set_page_config(layout="wide")
    st.title("ARTLib Studio — Step-By-Step Fuzzy ART Explorer (v0.3)")

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
        st.subheader("Settings")
        n = st.slider("n samples", 5, 200, 20)
        centers = st.slider("centers", 1, 6, 3)
        seed = st.number_input("seed", value=42)

        st.markdown("---")
        st.subheader("Fuzzy ART Parameters")
        rho = st.slider("vigilance (rho)", 0.0, 1.0, 0.85, step=0.01)
        alpha = st.slider("choice (alpha)", 0.0, 1.0, 0.0, step=0.01)
        beta = st.slider("learning rate (beta)", 0.0, 1.0, 1.0, step=0.01)

        if st.button("Generate dataset"):
            X, y = make_dataset(n_samples=n, centers=centers, seed=seed)
            st.session_state.X = X
            st.session_state.labels = None
            st.session_state.recorder = None
            st.session_state.current_event_index = -1

        if st.button("Run FuzzyART"):
            if st.session_state.X is None:
                st.warning("Generate dataset first")
            else:
                recorder = TraceRecorder()
                art = FuzzyART(rho=float(rho), alpha=float(alpha), beta=float(beta))
                instr = InstrumentedART(art, recorder=recorder)

                # Fit the data
                instr.fit(st.session_state.X)

                st.session_state.recorder = recorder
                st.session_state.art = art
                st.session_state.instrumented = instr
                st.session_state.labels = art.labels_
                st.session_state.current_event_index = 0 if len(recorder.events) > 0 else -1

        if st.button("Reset Explorer"):
            st.session_state.recorder = None
            st.session_state.art = None
            st.session_state.instrumented = None
            st.session_state.labels = None
            st.session_state.current_event_index = -1

    with col_main:
        st.subheader("Visualization")
        if st.session_state.recorder is None:
            st.info("Run FuzzyART to see the visualization.")
            if st.session_state.X is not None:
                # Just show the data
                fig, ax = plt.subplots(figsize=(6,4))
                ax.scatter(st.session_state.X[:,0], st.session_state.X[:,1], s=20, color="gray")
                ax.set_title("Input Data")
                st.pyplot(fig)
        else:
            rec = st.session_state.recorder
            fig = plot_state(st.session_state.X, st.session_state.labels, st.session_state.art, rec, st.session_state.current_event_index)
            st.pyplot(fig)
            st.caption("Category regions are approximate projections of Fuzzy ART learned expectations. They are not hard decision boundaries. Overlap can be normal. Actual category selection is determined by choice scores and vigilance match.")

            # Add Internals Inspector
            show_internals_inspector(rec, st.session_state.current_event_index)

    with col_right:
        st.subheader("Event Trace")
        if st.session_state.recorder is None:
            st.write("No trace available.")
        else:
            rec = st.session_state.recorder

            # Global Step Controls
            st.write("**Global Step Controls**")
            c1, c2 = st.columns([1, 1])
            if c1.button("<< Step Prev Event"):
                if st.session_state.current_event_index > 0:
                    st.session_state.current_event_index -= 1
            if c2.button("Step Next Event >>"):
                if st.session_state.current_event_index < len(rec.events) - 1:
                    st.session_state.current_event_index += 1

            curr_ev = rec.events[st.session_state.current_event_index] if st.session_state.current_event_index >= 0 else None
            if curr_ev:
                st.markdown(f"**Current Event ({st.session_state.current_event_index + 1}/{len(rec.events)})**")
                st.info(curr_ev.payload.get("explanation", curr_ev.type.value))
                with st.expander("Event Payload"):
                    st.json(curr_ev.payload)

            st.markdown("---")
            # Group events by sample
            sample_ids = sorted(list(set([e.payload.get("sample_index", -1) for e in rec.events if e.payload.get("sample_index", -1) != -1])))
            if sample_ids:
                selected_sample = st.selectbox("Inspect specific sample trace:", sample_ids, index=0)
                sample_events = [e for e in rec.events if e.payload.get("sample_index", -1) == selected_sample]

                st.write(f"**Trace for Sample {selected_sample}:**")
                for e in sample_events:
                    color = "blue" if e.type == EventType.RESONANCE else ("green" if e.type == EventType.CATEGORY_CREATED else "black")
                    st.markdown(f"- :{color}[{e.type.value}]: {e.payload.get('explanation', '')}")

if __name__ == "__main__":
    main()
