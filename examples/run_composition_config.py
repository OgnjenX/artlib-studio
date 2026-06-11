"""Load, run, and export a configured ART composition graph."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


STUDIO_ROOT = Path(__file__).resolve().parents[1]
ARTLIB_ROOT = STUDIO_ROOT.parent / "AdaptiveResonanceLib"
for path in (STUDIO_ROOT, ARTLIB_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from artlib_studio.composition.builder import build_graph_from_config
from artlib_studio.composition.config import load_graph_config
from artlib_studio.composition.signals import InputSignal
from artlib_studio.visualization.composition_trace import (
    summarize_bidirectional_step,
)


DATA = np.array(
    [[0.10, 0.12], [0.12, 0.14], [0.82, 0.85], [0.80, 0.83]],
    dtype=float,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = load_graph_config(args.config)
    graph = build_graph_from_config(config)
    input_module = next(
        module.id
        for module in config.modules
        if module.type == "adapter"
    )

    for sample_index, sample in enumerate(DATA):
        graph.step(
            [
                InputSignal(
                    source_module_id="environment",
                    target_module_id=input_module,
                    step_index=sample_index,
                    payload={"input": sample.tolist()},
                )
            ]
        )
        summary = summarize_bidirectional_step(
            graph.get_event_log(), sample_index
        )
        print(
            f"Sample {sample_index}: "
            f"categories={summary['selected_categories']} "
            f"result={summary['cross_module_result']}"
        )

    output_path = (
        Path(__file__).parent / "output" / f"{config.name}_trace.json"
    )
    graph.export_event_log_json(output_path)
    print(f"Graph state: {graph.get_global_state()}")
    print(f"Trace exported to {output_path}")


if __name__ == "__main__":
    main()
