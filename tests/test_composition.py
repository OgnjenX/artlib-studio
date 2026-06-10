from typing import Any, Dict, List

from artlib_studio.adapters.fuzzy_art import FuzzyARTAdapter
from artlib_studio.composition.edges import EdgeType, ModuleEdge
from artlib_studio.composition.events import CompositionEventType
from artlib_studio.composition.graph import ARTCompositionGraph
from artlib_studio.composition.module import AdapterARTModule, ComposableARTModule
from artlib_studio.composition.signals import (
    CompositionSignal,
    InputSignal,
    SelectedCategorySignal,
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
