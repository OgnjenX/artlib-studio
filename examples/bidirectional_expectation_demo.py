"""Run the simplified bidirectional ART expectation prototype."""
from __future__ import annotations

from functools import partial
import sys
from pathlib import Path

import numpy as np


STUDIO_ROOT = Path(__file__).resolve().parents[1]
ARTLIB_ROOT = STUDIO_ROOT.parent / "AdaptiveResonanceLib"
for path in (STUDIO_ROOT, ARTLIB_ROOT):
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)

from artlib_studio.adapters.fuzzy_art import FuzzyARTAdapter
from artlib_studio.composition.edges import EdgeType, ModuleEdge
from artlib_studio.composition.graph import ARTCompositionGraph
from artlib_studio.composition.module import AdapterARTModule
from artlib_studio.composition.signals import InputSignal
from artlib_studio.composition.transforms import (
    high_category_to_expectation,
    selected_category_to_one_hot,
)
from artlib_studio.visualization.composition_trace import (
    summarize_bidirectional_step,
)


def make_demo_data() -> np.ndarray:
    return np.array(
        [
            [0.10, 0.12],
            [0.12, 0.14],
            [0.82, 0.85],
            [0.80, 0.83],
            [0.11, 0.10],
            [0.84, 0.81],
        ],
        dtype=float,
    )


def build_bidirectional_graph(one_hot_size: int = 4) -> ARTCompositionGraph:
    graph = ARTCompositionGraph(max_settling_steps=5)
    graph.add_module(
        AdapterARTModule(
            "low_level_fuzzy_art",
            FuzzyARTAdapter(),
            {"rho": 0.85, "alpha": 0.001, "beta": 1.0},
        )
    )
    graph.add_module(
        AdapterARTModule(
            "high_level_fuzzy_art",
            FuzzyARTAdapter(),
            {"rho": 0.4, "alpha": 0.001, "beta": 1.0},
        )
    )
    graph.add_edge(
        ModuleEdge(
            "low_level_fuzzy_art",
            "high_level_fuzzy_art",
            EdgeType.BOTTOM_UP,
            transform=partial(
                selected_category_to_one_hot,
                vector_size=one_hot_size,
            ),
            transform_name=f"selected_category_to_one_hot[{one_hot_size}]",
        )
    )
    graph.add_edge(
        ModuleEdge(
            "high_level_fuzzy_art",
            "low_level_fuzzy_art",
            EdgeType.TOP_DOWN_EXPECTATION,
            transform=partial(
                high_category_to_expectation,
                association_memory=graph.association_memory,
            ),
            transform_name="high_category_to_expectation",
        )
    )
    return graph


def run_demo(data: np.ndarray | None = None) -> ARTCompositionGraph:
    graph = build_bidirectional_graph()
    samples = make_demo_data() if data is None else data
    for sample_index, sample in enumerate(samples):
        graph.step(
            [
                InputSignal(
                    source_module_id="environment",
                    target_module_id="low_level_fuzzy_art",
                    step_index=sample_index,
                    payload={"input": sample.tolist()},
                    summary=f"External sample {sample_index}.",
                )
            ]
        )
        summary = summarize_bidirectional_step(
            graph.get_event_log(),
            sample_index,
        )
        selected = summary["selected_categories"]
        expectation = summary["expectation"]
        result = summary["cross_module_result"]
        print(f"\nSample {sample_index}: {sample.tolist()}")
        print(
            "  low-level category = "
            f"{selected.get('low_level_fuzzy_art')}"
        )
        print(
            "  high-level category = "
            f"{selected.get('high_level_fuzzy_art')}"
        )
        if expectation:
            if expectation["matched"] is None:
                print(
                    "  High-level category has no learned expectation yet."
                )
            else:
                print(
                    "  high-level expects low-level categories "
                    f"{expectation['expected_category_ids']}"
                )
                print(
                    "  current low-level category = "
                    f"{expectation['current_low_level_category_id']}"
                )
        print(
            "  result = "
            f"{result['result'] if result else 'NO_EXPECTATION_RESULT'}"
        )
        print(f"  learned associations = {graph.association_memory.to_dict()}")
    return graph


def main() -> None:
    print("Bidirectional ART expectation prototype")
    print(
        "Expectations use learned category associations. This is not a full "
        "recurrent neural ART circuit."
    )
    graph = run_demo()
    print("\nFinal graph state:")
    print(graph.get_global_state())


if __name__ == "__main__":
    main()
