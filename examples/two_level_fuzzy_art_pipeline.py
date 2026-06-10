"""Run a feed-forward two-level Fuzzy ART composition experiment."""
from __future__ import annotations

from functools import partial

import numpy as np

from artlib_studio.adapters.fuzzy_art import FuzzyARTAdapter
from artlib_studio.composition.edges import EdgeType, ModuleEdge
from artlib_studio.composition.graph import ARTCompositionGraph
from artlib_studio.composition.module import AdapterARTModule
from artlib_studio.composition.signals import InputSignal
from artlib_studio.composition.transforms import selected_category_to_one_hot
from artlib_studio.visualization.composition_trace import summarize_graph_step


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


def build_two_level_graph(one_hot_size: int = 4) -> ARTCompositionGraph:
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
            {"rho": 0.75, "alpha": 0.001, "beta": 1.0},
        )
    )
    graph.add_edge(
        ModuleEdge(
            source_module_id="low_level_fuzzy_art",
            target_module_id="high_level_fuzzy_art",
            edge_type=EdgeType.BOTTOM_UP,
            transform=partial(
                selected_category_to_one_hot,
                vector_size=one_hot_size,
            ),
            transform_name=f"selected_category_to_one_hot[{one_hot_size}]",
        )
    )
    return graph


def run_pipeline(data: np.ndarray | None = None) -> ARTCompositionGraph:
    graph = build_two_level_graph()
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
        summary = summarize_graph_step(graph.get_event_log(), sample_index)
        low = summary["selected_categories"].get("low_level_fuzzy_art")
        high = summary["selected_categories"].get("high_level_fuzzy_art")
        vector = summary["signals"][0]["signal_payload"]["input"]
        print(f"\nSample {sample_index}: {sample.tolist()}")
        print(f"  low_level_fuzzy_art selected category {low}")
        print(f"  bottom-up one-hot signal: {vector}")
        print(f"  high_level_fuzzy_art selected category {high}")
        for event in summary["events"]:
            print(
                f"  {event['type']}"
                f" module={event.get('module_id')}"
                f" payload={event.get('payload', {})}"
            )
    return graph


def main() -> None:
    print("Feed-forward two-level Fuzzy ART pipeline")
    print(
        "The one-hot category code is an engineering simplification, "
        "not a full ART neural code."
    )
    graph = run_pipeline()
    print("\nFinal graph state:")
    print(graph.get_global_state())


if __name__ == "__main__":
    main()
