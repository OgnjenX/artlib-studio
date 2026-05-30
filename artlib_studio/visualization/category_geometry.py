"""Visualization for category geometries."""
import matplotlib.pyplot as plt
import numpy as np
from ..core.recorder import TraceRecorder

def plot_state(X, labels, model, adapter, recorder: TraceRecorder, current_event_index: int):
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

    # Plot category bounding boxes or regions
    from ..core.capabilities import Capability
    try:
        if adapter.supports(Capability.CATEGORY_BOXES_2D):
            bboxes = adapter.get_category_geometry_2d(model)
            for box in bboxes:
                rect = plt.Rectangle((box["x"], box["y"]), box["width"], box["height"],
                                   fill=False, edgecolor="C%d" % (box["id"] % 10), linewidth=2)
                ax_points.add_patch(rect)
                ax_points.text(box["x"], box["y"], f"Cat {box['id']}", color="C%d" % (box["id"] % 10), fontsize=9)
    except Exception:
        pass

    try:
        if adapter.supports(Capability.HYPERSPHERE_REGIONS_2D):
            spheres = adapter.get_category_geometry_2d(model)
            for s in spheres:
                circ = plt.Circle((s["x"], s["y"]), s["radius"], fill=False, edgecolor="C%d" % (s["id"] % 10), linewidth=2)
                ax_points.add_patch(circ)
                ax_points.text(s["x"], s["y"], f"Cat {s['id']}", color="C%d" % (s["id"] % 10), fontsize=9)
    except Exception:
        pass

    try:
        if adapter.supports(Capability.GAUSSIAN_REGIONS_2D):
            from matplotlib.patches import Ellipse
            gaussians = adapter.get_category_geometry_2d(model)
            for g in gaussians:
                # Plot 2-sigma ellipse (approx 95% confidence bounds)
                ell = Ellipse(xy=(g["x"], g["y"]),
                              width=g["sigma_x"] * 4,
                              height=g["sigma_y"] * 4,
                              fill=False, edgecolor="C%d" % (g["id"] % 10), linewidth=2, linestyle='--')
                ax_points.add_patch(ell)
                ax_points.text(g["x"], g["y"], f"Cat {g['id']}", color="C%d" % (g["id"] % 10), fontsize=9)
    except Exception:
        pass

    ax_points.set_title("Data and Approximate Category Expectations")
    ax_points.set_xlim(-0.05, 1.05)
    ax_points.set_ylim(-0.05, 1.05)
    ax_points.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    return fig
