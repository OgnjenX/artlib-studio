"""Compare loose and context-modulated Fuzzy ART vigilance."""
from __future__ import annotations

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
from artlib_studio.composition.events import CompositionEventType
from artlib_studio.composition.signals import InputSignal


DATA = np.array(
    [[0.10, 0.10], [0.18, 0.18], [0.26, 0.26], [0.34, 0.34]],
    dtype=float,
)


def run_modulatory_demo():
    config_path = Path(__file__).parent / "configs" / "modulatory_vigilance.yaml"
    strict_graph = build_graph_from_config(load_graph_config(config_path))

    baseline_config = load_graph_config(config_path)
    baseline_config.modules = [
        module
        for module in baseline_config.modules
        if module.type != "context"
    ]
    baseline_config.edges = [
        edge
        for edge in baseline_config.edges
        if edge.type != "MODULATORY"
    ]
    baseline_graph = build_graph_from_config(baseline_config)

    for sample_index, sample in enumerate(DATA):
        for label, graph in (
            ("loose", baseline_graph),
            ("strict", strict_graph),
        ):
            module = graph.get_module("low_level_fuzzy_art")
            base_rho_before = module.get_params()["rho"]
            graph.step(
                [
                    InputSignal(
                        source_module_id="environment",
                        target_module_id="low_level_fuzzy_art",
                        step_index=sample_index,
                        payload={"input": sample.tolist()},
                    )
                ]
            )
            state = module.get_state()
            step_events = [
                event
                for event in graph.get_event_log()
                if event.step_index == sample_index
                and event.module_id == "low_level_fuzzy_art"
            ]
            modulation_events = [
                event
                for event in step_events
                if event.type
                == CompositionEventType.MODULE_PARAMETER_MODULATED
            ]
            applied_rho = (
                modulation_events[-1].payload["after"]
                if modulation_events
                else base_rho_before
            )
            reset = any(
                event.type == CompositionEventType.MODULE_RESET
                for event in step_events
            )
            category_created = any(
                event.type == CompositionEventType.MODULE_CATEGORY_CREATED
                for event in step_events
            )
            print(
                f"{label} sample={sample_index} input={sample.tolist()}\n"
                f"  base_rho_before={base_rho_before}\n"
                f"  applied_rho_during_processing={applied_rho}\n"
                f"  selected_category={state['selected_category']}\n"
                f"  category_count={state['category_count']}\n"
                f"  reset={reset}\n"
                f"  category_created={category_created}\n"
                f"  restored_rho_after_step={state['params']['rho']}"
            )
    print(
        "Final category counts:",
        {
            "loose": baseline_graph.get_module(
                "low_level_fuzzy_art"
            ).get_state()["category_count"],
            "strict": strict_graph.get_module(
                "low_level_fuzzy_art"
            ).get_state()["category_count"],
        },
    )
    return baseline_graph, strict_graph


if __name__ == "__main__":
    run_modulatory_demo()
