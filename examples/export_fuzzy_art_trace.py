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
    X = np.array([
        [0.1, 0.2],
        [0.85, 0.9],
        [0.15, 0.18],
        [0.78, 0.79]
    ])

    recorder = TraceRecorder()
    adapter = get_adapter("fuzzy_art")
    params = {"rho": 0.85, "alpha": 0.0, "beta": 1.0}
    adapter.fit_with_trace(X, params, recorder=recorder)

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, "fuzzy_art_trace.json")
    with open(file_path, "w") as f:
        f.write(recorder.to_json())

    print(f"Trace successfully exported to: {file_path}")

if __name__ == "__main__":
    main()
