import json
from functools import partial
from typing import Any, Dict, List

from artlib_studio.adapters.fuzzy_art import FuzzyARTAdapter
from artlib_studio.composition.edges import EdgeType, ModuleEdge
from artlib_studio.composition.association_memory import AssociationMemory
from artlib_studio.composition.events import CompositionEventType
from artlib_studio.composition.graph import ARTCompositionGraph
from artlib_studio.composition.module import AdapterARTModule, ComposableARTModule
from artlib_studio.composition.signals import (
    CompositionSignal,
    InputSignal,
    ExpectationSignal,
    SelectedCategorySignal,
)
from artlib_studio.composition.transforms import (
    high_category_to_expectation,
    selected_category_to_one_hot,
    selected_category_to_scalar_vector,
)
from artlib_studio.core.events import EventType
from artlib_studio.visualization.composition_trace import (
    build_cross_module_resonance_table,
    build_expectation_table,
    build_module_timeline,
    build_signal_flow_table,
    summarize_graph_step,
    summarize_bidirectional_step,
)


class RelayModule(ComposableARTModule):
    def __init__(self, module_id: str):
        super().__init__(module_id)
        self.received: List[CompositionSignal] = []
        self.outputs: List[CompositionSignal] = []
        self.step_count = 0

    def receive(self, signal: CompositionSignal) -> None:
        self.received.append(signal)

    def step(self) -> None:
        self.step_count += 1
        pending = self.received
        self.received = []
        self.outputs = [
            SelectedCategorySignal(
                source_module_id=self.module_id,
                step_index=signal.step_index,
                payload={"category_id": 0},
                summary="Relay selected category 0.",
            )
            for signal in pending
        ]

    def get_state(self) -> Dict[str, Any]:
        return {"step_count": self.step_count}

    def get_output_signals(self) -> List[CompositionSignal]:
        outputs = self.outputs
        self.outputs = []
        return outputs

    def reset_runtime_state(self) -> None:
        self.received = []
        self.outputs = []
        self.step_count = 0

    def supports_expectation(self) -> bool:
        return False

    def explain_state(self) -> str:
        return f"Stepped {self.step_count} times."


def test_signal_creation():
    signal = InputSignal(
        source_module_id="external",
        target_module_id="art",
        step_index=3,
        payload={"input": [0.1, 0.2]},
        summary="A sample.",
    )
    assert signal.source_module_id == "external"
    assert signal.target_module_id == "art"
    assert signal.step_index == 3
    assert signal.payload["input"] == [0.1, 0.2]


def test_edge_creation_and_transmission():
    edge = ModuleEdge("a", "b", EdgeType.BOTTOM_UP, transform_name="identity")
    signal = SelectedCategorySignal(source_module_id="a")
    transmitted = edge.transmit(signal)
    assert edge.edge_type == EdgeType.BOTTOM_UP
    assert transmitted.target_module_id == "b"


def test_graph_add_module_and_edge():
    graph = ARTCompositionGraph()
    graph.add_module(RelayModule("a"))
    graph.add_module(RelayModule("b"))
    edge = ModuleEdge("a", "b", EdgeType.ASSOCIATIVE)
    graph.add_edge(edge)
    assert graph.get_module("a").module_id == "a"
    assert graph.edges == [edge]


def test_scheduler_delivers_signals_and_records_activity():
    graph = ARTCompositionGraph()
    source = RelayModule("source")
    target = RelayModule("target")
    graph.add_module(source)
    graph.add_module(target)
    graph.add_edge(ModuleEdge("source", "target", EdgeType.ASSOCIATIVE))

    settled = graph.step(
        [InputSignal("external", target_module_id="source", payload={"input": [1.0]})]
    )

    assert settled is True
    assert source.step_count == 1
    assert target.step_count == 1
    event_types = [event.type for event in graph.get_event_log()]
    assert CompositionEventType.SIGNAL_RECEIVED in event_types
    assert CompositionEventType.SIGNAL_SENT in event_types
    assert CompositionEventType.GRAPH_SETTLED in event_types


def test_adapter_module_emits_output_signals():
    module = AdapterARTModule("fuzzy", FuzzyARTAdapter())
    module.receive(
        InputSignal(
            "external",
            target_module_id="fuzzy",
            payload={"input": [0.1, 0.2]},
        )
    )
    module.step()
    outputs = module.get_output_signals()
    assert any(isinstance(signal, SelectedCategorySignal) for signal in outputs)
    assert module.get_state()["category_count"] == 1
    assert module.trace_recorder.events
    prepared = module.trace_recorder.events[0].payload["prepared_input"]
    assert prepared == [0.1, 0.2, 0.9, 0.8]

    module.receive(
        InputSignal(
            "external",
            target_module_id="fuzzy",
            payload={"input": [0.12, 0.22]},
        )
    )
    module.step()
    assert module.get_state()["sample_count"] == 2


def test_max_settling_steps_prevents_infinite_recurrent_loop():
    graph = ARTCompositionGraph(max_settling_steps=3)
    module = RelayModule("loop")
    graph.add_module(module)
    graph.add_edge(ModuleEdge("loop", "loop", EdgeType.ASSOCIATIVE))

    settled = graph.step(
        [InputSignal("external", target_module_id="loop", payload={"input": [1.0]})]
    )

    assert settled is False
    assert module.step_count == 3
    assert graph.get_event_log()[-1].type == CompositionEventType.GRAPH_FAILED_TO_SETTLE


def _build_two_level_test_graph():
    graph = ARTCompositionGraph()
    low = AdapterARTModule("low", FuzzyARTAdapter(), {"rho": 0.8})
    high = AdapterARTModule("high", FuzzyARTAdapter(), {"rho": 0.7})
    graph.add_module(low)
    graph.add_module(high)
    graph.add_edge(
        ModuleEdge(
            "low",
            "high",
            EdgeType.BOTTOM_UP,
            transform=partial(selected_category_to_one_hot, vector_size=4),
            transform_name="selected_category_to_one_hot[4]",
        )
    )
    return graph, low, high


def test_two_adapter_modules_process_transformed_signal():
    graph, low, high = _build_two_level_test_graph()
    settled = graph.step(
        [
            InputSignal(
                "environment",
                target_module_id="low",
                payload={"input": [0.1, 0.2]},
            )
        ]
    )

    assert settled is True
    assert low.get_state()["selected_category"] == 0
    assert high.get_state()["selected_category"] == 0
    assert high.get_state()["sample_count"] == 1
    high_input = high.trace_recorder.events[0].payload["input"]
    assert high_input == [1.0, 0.0, 0.0, 0.0]

    selected_modules = {
        event.module_id
        for event in graph.get_event_log()
        if event.type == CompositionEventType.MODULE_SELECTED_CATEGORY
    }
    assert selected_modules == {"low", "high"}


def test_created_category_is_emitted_as_final_selection():
    module = AdapterARTModule("fuzzy", FuzzyARTAdapter(), {"rho": 0.9})
    selections = []
    for value in ([0.1, 0.1], [0.9, 0.9]):
        module.receive(
            InputSignal(
                "environment",
                target_module_id="fuzzy",
                payload={"input": value},
            )
        )
        module.step()
        selections.extend(
            signal.payload["category_id"]
            for signal in module.get_output_signals()
            if isinstance(signal, SelectedCategorySignal)
        )

    assert selections == [0, 1]
    assert any(
        event.type == EventType.RESET for event in module.trace_recorder.events
    )


def test_category_transforms_create_valid_input_vectors():
    selected = SelectedCategorySignal(
        source_module_id="low",
        step_index=2,
        payload={"category_id": 2},
    )
    one_hot = selected_category_to_one_hot(selected, vector_size=4)
    scalar = selected_category_to_scalar_vector(selected, max_category_id=4)

    assert isinstance(one_hot, InputSignal)
    assert one_hot.payload["input"] == [0.0, 0.0, 1.0, 0.0]
    assert scalar.payload["input"] == [0.5]
    assert selected_category_to_one_hot(InputSignal("low"), vector_size=4) is None


def test_composition_trace_helpers_and_json_export(tmp_path):
    graph, _, _ = _build_two_level_test_graph()
    graph.step(
        [
            InputSignal(
                "environment",
                target_module_id="low",
                payload={"input": [0.1, 0.2]},
            )
        ]
    )

    trace = graph.get_event_log()
    timeline = build_module_timeline(trace)
    flow = build_signal_flow_table(trace)
    summary = summarize_graph_step(trace, 0)
    output_path = graph.export_event_log_json(tmp_path / "trace.json")
    exported = json.loads(output_path.read_text(encoding="utf-8"))

    assert {row["module_id"] for row in timeline} == {"low", "high"}
    assert flow[0]["source_module_id"] == "low"
    assert flow[0]["target_module_id"] == "high"
    assert flow[0]["signal_payload"]["input"] == [1.0, 0.0, 0.0, 0.0]
    assert summary["selected_categories"] == {"low": 0, "high": 0}
    assert summary["settled"] is True
    assert exported[0]["external_sample_index"] == 0
    assert any(
        event["type"] == CompositionEventType.MODULE_CATEGORY_CREATED.value
        for event in exported
    )


def test_expectation_signal_and_association_memory():
    signal = ExpectationSignal(
        source_module_id="high",
        target_module_id="low",
        expected_category_id=2,
        confidence=0.8,
        payload={"expected_category_ids": [2, 3]},
        explanation="Expected low-level categories 2 or 3.",
    )
    memory = AssociationMemory()
    memory.record_pair(1, 2)
    memory.record_pair(1, 3)

    assert signal.expected_category_ids == [2, 3]
    assert signal.confidence == 0.8
    assert memory.get_expected_low_level_categories(1) == {2, 3}
    assert memory.match_expectation(1, 2) is True
    assert memory.match_expectation(1, 4) is False


def _build_bidirectional_test_graph(expected_low_category):
    graph, low, high = _build_two_level_test_graph()
    graph.add_edge(
        ModuleEdge(
            "high",
            "low",
            EdgeType.TOP_DOWN_EXPECTATION,
            transform=partial(
                high_category_to_expectation,
                association_memory=graph.association_memory,
            ),
            transform_name="high_category_to_expectation",
        )
    )
    graph.association_memory.record_pair(0, expected_low_category)
    return graph, low, high


def test_scheduler_emits_cross_module_resonance_for_match():
    graph, _, _ = _build_bidirectional_test_graph(expected_low_category=0)
    graph.step(
        [InputSignal("environment", target_module_id="low", payload={"input": [0.1, 0.2]})]
    )
    event_types = [event.type for event in graph.get_event_log()]

    assert CompositionEventType.EXPECTATION_SENT in event_types
    assert CompositionEventType.EXPECTATION_RECEIVED in event_types
    assert CompositionEventType.EXPECTATION_MATCHED in event_types
    assert CompositionEventType.CROSS_MODULE_RESONANCE in event_types


def test_scheduler_emits_expectation_unknown_when_no_prior_association():
    graph, _, _ = _build_two_level_test_graph()
    graph.add_edge(
        ModuleEdge(
            "high",
            "low",
            EdgeType.TOP_DOWN_EXPECTATION,
            transform=partial(
                high_category_to_expectation,
                association_memory=graph.association_memory,
            ),
            transform_name="high_category_to_expectation",
        )
    )

    graph.step(
        [InputSignal("environment", target_module_id="low", payload={"input": [0.1, 0.2]})]
    )
    event_types = [event.type for event in graph.get_event_log()]
    expectation_rows = build_expectation_table(graph.get_event_log())
    resonance_rows = build_cross_module_resonance_table(graph.get_event_log())
    summary = summarize_bidirectional_step(graph.get_event_log(), 0)

    assert CompositionEventType.EXPECTATION_UNAVAILABLE in event_types
    assert CompositionEventType.CROSS_MODULE_EXPECTATION_UNKNOWN in event_types
    assert expectation_rows[-1]["matched"] is None
    assert resonance_rows[-1]["result"] == "CROSS_MODULE_EXPECTATION_UNKNOWN"
    assert summary["cross_module_result"]["result"] == "CROSS_MODULE_EXPECTATION_UNKNOWN"


def test_scheduler_emits_cross_module_mismatch_and_exports_json(tmp_path):
    graph, _, _ = _build_bidirectional_test_graph(expected_low_category=3)
    graph.step(
        [InputSignal("environment", target_module_id="low", payload={"input": [0.1, 0.2]})]
    )
    event_types = [event.type for event in graph.get_event_log()]
    output_path = graph.export_event_log_json(tmp_path / "bidirectional.json")
    exported = json.loads(output_path.read_text(encoding="utf-8"))

    assert CompositionEventType.EXPECTATION_MISMATCHED in event_types
    assert CompositionEventType.CROSS_MODULE_MISMATCH in event_types
    assert any(event["type"] == "EXPECTATION_SENT" for event in exported)

    expectation_rows = build_expectation_table(graph.get_event_log())
    resonance_rows = build_cross_module_resonance_table(graph.get_event_log())
    summary = summarize_bidirectional_step(graph.get_event_log(), 0)
    assert expectation_rows[-1]["matched"] is False
    assert resonance_rows[-1]["result"] == "CROSS_MODULE_MISMATCH"
    assert summary["cross_module_result"]["matched"] is False
