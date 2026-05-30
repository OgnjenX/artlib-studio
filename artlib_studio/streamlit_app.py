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

# Make AdaptiveResonanceLib importable when running from the repo
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARL_PATH = os.path.join(ROOT, "AdaptiveResonanceLib")
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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax_points, ax_timeline = axes

    # Panel A & B: Data points and categories
    if labels is None:
        ax_points.scatter(X[:, 0], X[:, 1], s=10, color="gray")
    else:
        # color by label
        for k in np.unique(labels):
            mask = labels == k
            ax_points.scatter(X[mask, 0], X[mask, 1], s=10, label=f"c{k}")

    # plot category bounding boxes if available (2D only)
    try:
        if hasattr(art, "get_bounding_boxes") and art.n_clusters > 0:
            bboxes = art.get_bounding_boxes(n=2)
            for idx, bbox in enumerate(bboxes):
                ref, widths = bbox
                rect = plt.Rectangle((ref[0], ref[1]), widths[0], widths[1], fill=False, edgecolor="C%d" % (idx % 10))
                ax_points.add_patch(rect)
    except Exception:
        pass

    ax_points.set_title("Data and categories")
    ax_points.set_xlim(-0.05, 1.05)
    ax_points.set_ylim(-0.05, 1.05)

    # Panel F: timeline
    events = recorder.events
    y = 0
    ax_timeline.axis('off')
    for i, ev in enumerate(events):
        txt = f"{i}: {ev.type.value} {ev.payload}"
        color = "black"
        if i == current_event_index:
            color = "red"
        ax_timeline.text(0, 1 - (i * 0.06), txt, color=color, fontsize=8, family='monospace')

    plt.tight_layout()
    return fig


def main():
    st.title("ARTLib Studio — Execution Explorer (v0.1)")

    if "recorder" not in st.session_state:
        st.session_state.recorder = None
        st.session_state.art = None
        st.session_state.instrumented = None
        st.session_state.X = None
        st.session_state.labels = None
        st.session_state.current_event_index = -1

    col1, col2 = st.columns([1, 2])
    with col1:
        n = st.slider("n samples", 20, 1000, 200)
        centers = st.slider("centers", 1, 6, 3)
        seed = st.number_input("seed", value=0)
        rho = st.slider("vigilance (rho)", 0.0, 1.0, 0.85)
        alpha = st.number_input("alpha (choice)", value=0.0)
        beta = st.number_input("beta (learn)", value=1.0)

        if st.button("Generate dataset"):
            X, y = make_dataset(n_samples=n, centers=centers, seed=seed)
            st.session_state.X = X
            st.session_state.labels = None
            st.session_state.recorder = None

        if st.button("Run instrumented FuzzyART"):
            if st.session_state.X is None:
                st.warning("Generate dataset first")
            else:
                recorder = TraceRecorder()
                art = FuzzyART(rho=float(rho), alpha=float(alpha), beta=float(beta))
                instr = InstrumentedART(art, recorder=recorder)
                # prepare data using the model helper (complement code + normalize)
                Xp = instr.prepare_data(st.session_state.X)
                # step through samples and record events
                labels = -np.ones((Xp.shape[0],), dtype=int)
                for i, x in enumerate(Xp):
                    c = instr.step_fit(x)
                    labels[i] = c
                st.session_state.recorder = recorder
                st.session_state.art = art
                st.session_state.instrumented = instr
                st.session_state.labels = labels
                st.session_state.current_event_index = 0 if len(recorder.events) > 0 else -1

    with col2:
        st.header("Execution Explorer")
        if st.session_state.recorder is None:
            st.info("No execution recorded yet. Generate dataset and run instrumented FuzzyART.")
            return

        rec = st.session_state.recorder
        idx = st.session_state.current_event_index

        # Controls
        c1, c2, c3 = st.columns([1, 1, 1])
        if c1.button("<< Prev"):
            ev = rec.step_backward()
            if ev is not None:
                st.session_state.current_event_index = rec.index
        if c2.button("Next >>"):
            ev = rec.step_forward()
            if ev is not None:
                st.session_state.current_event_index = rec.index
        if c3.button("Replay"):
            # simple replay: iterate and advance index with small pause
            for i, ev in enumerate(rec.events):
                st.session_state.current_event_index = i
                st.experimental_rerun()

        # show current event detail
        ev = rec.current()
        if ev is not None:
            st.subheader(f"Event {rec.index}: {ev.type.value}")
            st.json({"timestamp": ev.timestamp, "payload": ev.payload})

        # render state plot
        fig = plot_state(st.session_state.X, st.session_state.labels, st.session_state.art, rec, st.session_state.current_event_index)
        st.pyplot(fig)


if __name__ == "__main__":
    main()

