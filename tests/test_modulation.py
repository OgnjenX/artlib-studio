import json
from pathlib import Path

from artlib_studio.composition.builder import build_graph_from_config
from artlib_studio.composition.config import load_graph_config
from artlib_studio.composition.context import ContextModule, ContextRule
from artlib_studio.composition.events import CompositionEventType
from artlib_studio.composition.signals import InputSignal, ModulatorySignal
from artlib_studio.visualization.composition_trace import (
    build_modulation_table,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "configs"
    / "modulatory_vigilance.yaml"
)


def test_modulatory_signal_and_context_module():
    signal = ModulatorySignal(
        "context",
        target_param="rho",
        mode="set",
        value=0.95,
        duration="current_step",
    )
    context = ContextModule(
        "context",
        [ContextRule("strict", value=0.95)],
    )
    outputs = context.emit_modulations(0)
    assert signal.target_param == "rho"
    assert outputs[0].value == 0.95


def test_context_config_builds_applies_and_restores_modulation():
    graph = build_graph_from_config(load_graph_config(CONFIG_PATH))
    module = graph.get_module("low_level_fuzzy_art")
    baseline_rho = module.get_params()["rho"]
    graph.step(
        [
            InputSignal(
                "environment",
                target_module_id="low_level_fuzzy_art",
                payload={"input": [0.1, 0.1]},
            )
        ]
    )
    event_types = [event.type for event in graph.get_event_log()]
    assert CompositionEventType.MODULATION_SENT in event_types
    assert CompositionEventType.MODULATION_RECEIVED in event_types
    assert CompositionEventType.MODULE_PARAMETER_MODULATED in event_types
    assert CompositionEventType.MODULE_PARAMETER_RESTORED in event_types
    assert module.get_params()["rho"] == baseline_rho


def test_current_step_modulation_occurs_before_input_processing():
    graph = build_graph_from_config(load_graph_config(CONFIG_PATH))
    graph.step(
        [
            InputSignal(
                "environment",
                target_module_id="low_level_fuzzy_art",
                payload={"input": [0.1, 0.1]},
            )
        ]
    )
    events = [
        event for event in graph.get_event_log() if event.step_index == 0
    ]

    def first_index(event_type, module_id=None):
        for index, event in enumerate(events):
            if event.type == event_type and (
                module_id is None or event.module_id == module_id
            ):
                return index
        raise AssertionError(f"Missing {event_type} for {module_id}")

    module_id = "low_level_fuzzy_art"
    modulation_sent = first_index(
        CompositionEventType.MODULATION_SENT,
        "strict_context",
    )
    modulation_received = first_index(
        CompositionEventType.MODULATION_RECEIVED,
        module_id,
    )
    modulated = first_index(
        CompositionEventType.MODULE_PARAMETER_MODULATED,
        module_id,
    )
    received_input = first_index(
        CompositionEventType.MODULE_RECEIVED_INPUT,
        module_id,
    )
    category_event = min(
        first_index(event_type, module_id)
        for event_type in (
            CompositionEventType.MODULE_SELECTED_CATEGORY,
            CompositionEventType.MODULE_CATEGORY_CREATED,
            CompositionEventType.MODULE_RESONATED,
        )
        if any(
            event.type == event_type and event.module_id == module_id
            for event in events
        )
    )
    restored = first_index(
        CompositionEventType.MODULE_PARAMETER_RESTORED,
        module_id,
    )
    settled = first_index(CompositionEventType.GRAPH_SETTLED)

    assert (
        modulation_sent
        < modulation_received
        < modulated
        < received_input
        < category_event
        < restored
        < settled
    )
    assert events[modulated].payload["after"] == 0.95
    assert events[restored].payload["restored_value"] == 0.6


def test_persistent_modulation_remains_active():
    config = load_graph_config(CONFIG_PATH)
    config.modules[1].rules[0].duration = "persistent"
    graph = build_graph_from_config(config)
    graph.step(
        [
            InputSignal(
                "environment",
                target_module_id="low_level_fuzzy_art",
                payload={"input": [0.1, 0.1]},
            )
        ]
    )
    assert graph.get_module("low_level_fuzzy_art").get_params()["rho"] == 0.95


def test_strict_vigilance_changes_category_granularity():
    strict_graph = build_graph_from_config(load_graph_config(CONFIG_PATH))
    loose_config = load_graph_config(CONFIG_PATH)
    loose_config.modules = [
        module for module in loose_config.modules if module.type == "adapter"
    ]
    loose_config.edges = []
    loose_graph = build_graph_from_config(loose_config)
    data = [[0.10, 0.10], [0.18, 0.18], [0.26, 0.26], [0.34, 0.34]]
    for index, sample in enumerate(data):
        for graph in (loose_graph, strict_graph):
            graph.step(
                [
                    InputSignal(
                        "environment",
                        target_module_id="low_level_fuzzy_art",
                        step_index=index,
                        payload={"input": sample},
                    )
                ]
            )
    loose_count = loose_graph.get_module("low_level_fuzzy_art").get_state()[
        "category_count"
    ]
    strict_count = strict_graph.get_module("low_level_fuzzy_art").get_state()[
        "category_count"
    ]
    strict_events = strict_graph.get_event_log()
    assert strict_count > loose_count
    assert any(
        event.type == CompositionEventType.MODULE_PARAMETER_MODULATED
        for event in strict_events
    )


def test_modulation_trace_table_and_json_export(tmp_path):
    graph = build_graph_from_config(load_graph_config(CONFIG_PATH))
    graph.step(
        [
            InputSignal(
                "environment",
                target_module_id="low_level_fuzzy_art",
                payload={"input": [0.1, 0.1]},
            )
        ]
    )
    rows = build_modulation_table(graph.get_event_log())
    assert [row["event_type"] for row in rows] == [
        CompositionEventType.MODULATION_SENT.value,
        CompositionEventType.MODULATION_RECEIVED.value,
        CompositionEventType.MODULE_PARAMETER_MODULATED.value,
        CompositionEventType.MODULE_PARAMETER_RESTORED.value,
    ]
    assert rows[-1]["restored_value"] == 0.6

    output_path = graph.export_event_log_json(tmp_path / "trace.json")
    event_types = {
        event["type"] for event in json.loads(output_path.read_text())
    }
    assert {
        CompositionEventType.MODULATION_SENT.value,
        CompositionEventType.MODULATION_RECEIVED.value,
        CompositionEventType.MODULE_PARAMETER_MODULATED.value,
        CompositionEventType.MODULE_PARAMETER_RESTORED.value,
        CompositionEventType.MODULE_SELECTED_CATEGORY.value,
        CompositionEventType.MODULE_CATEGORY_CREATED.value,
    } <= event_types
