import matplotlib.pyplot as plt
import numpy as np

from ..core.events import EventType
from .timeline_state import build_timeline_state

def render_process_visualization(
    X,
    labels,
    adapter,
    model,
    trace,
    current_event_index,
    current_sample_index=None,
    show_future_faintly=False
):
    fig, ax = plt.subplots(figsize=(6, 4))

    t_state = build_timeline_state(X, trace, current_event_index)

    if t_state is None:
        ax.set_title("No timeline state available")
        return fig

    # Determine event state
    events = trace.events if hasattr(trace, 'events') else trace
    current_event = t_state["current_event"]
    s_idx = t_state["current_sample_index"]

    processed = t_state["processed_sample_indices"]
    future = t_state["future_sample_indices"]

    # Plot base data - show only processed unless 'show_future_faintly' is set
    if labels is None:
        if processed:
            ax.scatter(X[processed, 0], X[processed, 1], s=20, color="gray", zorder=1)
    else:
        for k in np.unique(labels):
            mask = labels == k
            # only process samples in mask that are in processed
            p_mask = [i for i in processed if mask[i]]
            if p_mask:
                ax.scatter(X[p_mask, 0], X[p_mask, 1], s=20, label=f"c{k}", zorder=1)

    if future and show_future_faintly:
        ax.scatter(X[future, 0], X[future, 1], s=20, color="lightgray", alpha=0.3, zorder=0, label="Future")

    # Temporarily override model weights to historical if possible
    historical_weights = t_state["historical_weights"]
    categories_existing = set(t_state["categories_existing_so_far"])

    old_W = getattr(model, "W", None)
    if old_W is not None and historical_weights:
        try:
            max_c = max(categories_existing) if categories_existing else -1
            w_hist = []
            for c in range(max_c + 1):
                if c in historical_weights:
                    w_hist.append(np.array(historical_weights[c]))
                elif c < len(old_W):
                    w_hist.append(old_W[c])
            model.W = w_hist
        except Exception:
            pass

    # Plot category regions (using existing adapter functions if possible)
    from ..core.capabilities import Capability

    # helper internal function to try plotting geometries
    def draw_geometries():
        try:
            if adapter.supports(Capability.CATEGORY_BOXES_2D):
                bboxes = adapter.get_category_geometry_2d(model)
                for box in bboxes:
                    if box["id"] not in categories_existing: continue
                    rect = plt.Rectangle((box["x"], box["y"]), box["width"], box["height"],
                                    fill=False, edgecolor="C%d" % (box["id"] % 10), linewidth=2, alpha=0.5, zorder=2)
                    ax.add_patch(rect)
                    ax.text(box["x"], box["y"], f"Cat {box['id']}", color="C%d" % (box["id"] % 10), fontsize=9, zorder=3)
        except Exception:
            pass

        try:
            if adapter.supports(Capability.HYPERSPHERE_REGIONS_2D):
                spheres = adapter.get_category_geometry_2d(model)
                for s in spheres:
                    if s["id"] not in categories_existing: continue
                    circ = plt.Circle((s["x"], s["y"]), s["radius"], fill=False, edgecolor="C%d" % (s["id"] % 10), linewidth=2, alpha=0.5, zorder=2)
                    ax.add_patch(circ)
                    ax.text(s["x"], s["y"], f"Cat {s['id']}", color="C%d" % (s["id"] % 10), fontsize=9, zorder=3)
        except Exception:
            pass

        try:
            if adapter.supports(Capability.GAUSSIAN_REGIONS_2D):
                from matplotlib.patches import Ellipse
                gaussians = adapter.get_category_geometry_2d(model)
                for g in gaussians:
                    if g["id"] not in categories_existing: continue
                    ell = Ellipse(xy=(g["x"], g["y"]),
                                width=g["sigma_x"] * 4,
                                height=g["sigma_y"] * 4,
                                fill=False, edgecolor="C%d" % (g["id"] % 10), linewidth=2, linestyle='--', alpha=0.5, zorder=2)
                    ax.add_patch(ell)
                    ax.text(g["x"], g["y"], f"Cat {g['id']}", color="C%d" % (g["id"] % 10), fontsize=9, zorder=3)
        except Exception:
            pass

    draw_geometries()

    # Restore model weights
    if old_W is not None:
        model.W = old_W

    title = "ART Process"

    if s_idx is not None and s_idx >= 0:
        # Highlight input
        ax.scatter(X[s_idx, 0], X[s_idx, 1], s=150, facecolors='none', edgecolors='red', linewidths=2, label="Current Input", zorder=4)
        title = f"Processing Sample {s_idx}"

        if current_event is not None:
            c_id = current_event.payload.get("category_id", None)
            if c_id is not None:
                # Find category center so we can draw an arrow
                # For Fuzzy ART bounds, let's extract them
                cat_center = None
                try:
                    # just roughly estimate center
                    geom = adapter.get_category_geometry_2d(model)
                    for g in geom:
                        if g["id"] == c_id:
                            if "width" in g:
                                cat_center = (g["x"] + g["width"]/2, g["y"] + g["height"]/2)
                            elif "radius" in g:
                                cat_center = (g["x"], g["y"])
                except Exception:
                    pass

            etype = current_event.type
            if etype == EventType.INPUT_RECEIVED:
                title += " — Input Received"
            elif etype == EventType.CATEGORY_EVALUATED:
                title += " — Category Evaluation"
            elif etype == EventType.CATEGORY_SELECTED:
                title += f" — Selected Category {c_id}"
                if cat_center:
                    ax.annotate("", xy=cat_center, xytext=(X[s_idx, 0], X[s_idx, 1]), arrowprops=dict(arrowstyle="->", color="orange", lw=2))
            elif etype == EventType.MATCH_TEST:
                passed = current_event.payload.get("passed", False)
                title += f" — Match {'Passed' if passed else 'Failed'} for Category {c_id}"
            elif etype == EventType.RESET:
                reset_cats = current_event.payload.get("reset_categories", [])
                title += f" — Reset Category {c_id}"
            elif etype == EventType.RESONANCE:
                title += f" — Resonance with Category {c_id}"
                if cat_center:
                    ax.annotate("", xy=cat_center, xytext=(X[s_idx, 0], X[s_idx, 1]), arrowprops=dict(arrowstyle="->", color="green", lw=3))
            elif etype == EventType.LEARNING:
                title += f" — Learning Category {c_id}"
            elif etype == EventType.CATEGORY_CREATED:
                created_id = current_event.payload.get("created_index", current_event.payload.get("category_id", -1))
                title += f" — New Category {created_id} Created"
                # Mark created category
                if created_id != -1:
                    # Temporary override again just for this marker if needed
                    if old_W is not None and historical_weights:
                        try:
                            model.W = w_hist
                        except Exception: pass
                    try:
                        geom = adapter.get_category_geometry_2d(model)
                        for g in geom:
                            if g["id"] == created_id:
                                cx, cy = g["x"], g["y"]
                                ax.scatter(cx, cy, s=200, marker='*', color='gold', zorder=5, label="New Category")
                                ax.text(cx, cy, "NEW", color='red')
                    except Exception:
                        pass
                    if old_W is not None:
                        model.W = old_W

    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    # limit legend
    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')

    plt.tight_layout()
    return fig
