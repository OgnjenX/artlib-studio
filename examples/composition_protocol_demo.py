"""Protocol-only demonstration of signal delivery between ART modules."""

from artlib_studio.adapters.fuzzy_art import FuzzyARTAdapter
from artlib_studio.composition.edges import EdgeType, ModuleEdge
from artlib_studio.composition.graph import ARTCompositionGraph
from artlib_studio.composition.module import AdapterARTModule
from artlib_studio.composition.signals import InputSignal


def main() -> None:
    print(
        "Protocol-only demo: upper_art receives a selected-category signal, "
        "but AdapterARTModule does not yet process non-Input signals."
    )
    graph = ARTCompositionGraph(max_settling_steps=5)
    lower = AdapterARTModule("lower_art", FuzzyARTAdapter(), {"rho": 0.8})
    upper = AdapterARTModule("upper_art", FuzzyARTAdapter(), {"rho": 0.7})
    graph.add_module(lower)
    graph.add_module(upper)

    graph.add_edge(
        ModuleEdge(
            source_module_id="lower_art",
            target_module_id="upper_art",
            edge_type=EdgeType.BOTTOM_UP,
            transform_name="identity",
        )
    )

    graph.step(
        [
            InputSignal(
                source_module_id="environment",
                target_module_id="lower_art",
                payload={"input": [0.15, 0.2]},
                summary="Demo sample for the lower ART module.",
            )
        ]
    )

    print("Global state:")
    print(graph.get_global_state())
    print(
        "\nThis is not a meaningful two-level ART hierarchy: upper_art remains "
        "unprocessed because expectation/modulation and category-to-input "
        "transforms are not implemented."
    )
    print("\nComposition events:")
    for event in graph.get_event_log():
        print(event.to_dict())
    print("\nNative lower-module trace events:")
    for event in lower.trace_recorder.events:
        print(event.to_dict())


if __name__ == "__main__":
    main()
