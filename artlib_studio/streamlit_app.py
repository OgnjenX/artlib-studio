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

    ax_points.set_title("Data and Categories (Final State Approximation)")
    ax_points.set_xlim(-0.05, 1.05)
    ax_points.set_ylim(-0.05, 1.05)
    ax_points.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    return fig


def main():
    st.set_page_config(layout="wide")
    st.title("ARTLib Studio — Step-By-Step Fuzzy ART Explorer (v0.2)")

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
