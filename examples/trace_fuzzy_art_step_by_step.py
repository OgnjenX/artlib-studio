import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARL_PATH = os.path.join(ROOT, "AdaptiveResonanceLib")
STUDIO_PATH = os.path.join(ROOT, "artlib-studio")
if ARL_PATH not in sys.path:
    sys.path.insert(0, ARL_PATH)
if STUDIO_PATH not in sys.path:
    sys.path.insert(0, STUDIO_PATH)

from artlib_studio.core.registry import get_adapter
from artlib_studio.core.recorder import TraceRecorder

def main():
    # Tiny 2D synthetic dataset
    X = np.array([
        [0.1, 0.2],
        [0.85, 0.9],
        [0.15, 0.18],
        [0.78, 0.79]
    ])

    recorder = TraceRecorder()
    adapter = get_adapter("fuzzy_art")
    params = {"rho": 0.85, "alpha": 0.0, "beta": 1.0}

    print("Fuzzy ART Training Trace\n" + "="*24 + "\n")

    adapter.fit_with_trace(X, params, recorder=recorder)

    # Group events by sample
    grouped = {}
    for event in recorder.events:
        idx = event.payload.get("sample_index", -1)
        if idx not in grouped:
            grouped[idx] = []
        grouped[idx].append(event)

    for sample_idx, events in grouped.items():
        if sample_idx == -1:
            continue
        # Convert to normal floats for cleaner printing
        x_val = [round(float(v), 2) for v in X[sample_idx]]
        print(f"Sample {sample_idx}: {x_val}")
        for i, ev in enumerate(events, 1):
            explanation = adapter.explain_event(ev)
            print(f"  {i}. {explanation}")
        print()

if __name__ == "__main__":
    main()
