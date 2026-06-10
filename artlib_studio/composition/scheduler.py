"""Bounded discrete scheduler for ART composition graphs."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, TYPE_CHECKING

from .events import CompositionEvent, CompositionEventType
from .signals import (
    CompositionSignal,
    LearningSignal,
    ResetSignal,
    ResonanceSignal,
    SelectedCategorySignal,
)

if TYPE_CHECKING:
    from .graph import ARTCompositionGraph


class DiscreteScheduler:
    def __init__(self, max_settling_steps: int = 5):
        if max_settling_steps < 1:
            raise ValueError("max_settling_steps must be at least 1")
        self.max_settling_steps = max_settling_steps
        self.step_index = 0

    def run(
        self,
        graph: "ARTCompositionGraph",
        input_signals: Iterable[CompositionSignal],
    ) -> bool:
        pending = list(input_signals)
        for settling_step in range(self.max_settling_steps):
            if not pending:
                graph._record(
                    CompositionEvent(
                        CompositionEventType.GRAPH_SETTLED,
                        self.step_index,
                        payload={"settling_steps": settling_step},
                    )
                )
                self.step_index += 1
                return True

            recipients: Dict[str, List[CompositionSignal]] = defaultdict(list)
            for signal in pending:
                targets = (
                    [signal.target_module_id]
                    if signal.target_module_id is not None
                    else list(graph.modules)
                )
                for target in targets:
                    module = graph.get_module(target)
                    delivered = signal.for_target(target)
                    module.receive(delivered)
                    recipients[target].append(delivered)
                    graph._record(
                        CompositionEvent(
                            CompositionEventType.SIGNAL_RECEIVED,
                            self.step_index,
                            module_id=target,
                            payload={"signal_type": type(delivered).__name__},
                        )
                    )
                    if delivered.__class__.__name__ == "InputSignal":
                        graph._record(
                            CompositionEvent(
                                CompositionEventType.MODULE_RECEIVED_INPUT,
                                self.step_index,
                                module_id=target,
                                payload=dict(delivered.payload),
                            )
                        )

            emitted: List[CompositionSignal] = []
            for module_id in recipients:
                module = graph.get_module(module_id)
                module.step()
                outputs = module.get_output_signals()
                emitted.extend(outputs)
                self._record_module_outputs(graph, module_id, outputs)

            pending = []
            for signal in emitted:
                for edge in graph.outgoing_edges(signal.source_module_id):
                    transmitted = edge.transmit(signal)
                    pending.append(transmitted)
                    graph._record(
                        CompositionEvent(
                            CompositionEventType.SIGNAL_SENT,
                            self.step_index,
                            module_id=signal.source_module_id,
                            payload={
                                "target_module_id": edge.target_module_id,
                                "edge_type": edge.edge_type.value,
                                "signal_type": type(transmitted).__name__,
                            },
                        )
                    )

        graph._record(
            CompositionEvent(
                CompositionEventType.GRAPH_FAILED_TO_SETTLE,
                self.step_index,
                payload={"max_settling_steps": self.max_settling_steps},
            )
        )
        self.step_index += 1
        return False

    def _record_module_outputs(
        self,
        graph: "ARTCompositionGraph",
        module_id: str,
        outputs: List[CompositionSignal],
    ) -> None:
        event_types = {
            SelectedCategorySignal: CompositionEventType.MODULE_SELECTED_CATEGORY,
            ResonanceSignal: CompositionEventType.MODULE_RESONATED,
            ResetSignal: CompositionEventType.MODULE_RESET,
            LearningSignal: CompositionEventType.MODULE_LEARNED,
        }
        for signal in outputs:
            event_type = event_types.get(type(signal))
            if event_type is not None:
                graph._record(
                    CompositionEvent(
                        event_type,
                        self.step_index,
                        module_id=module_id,
                        payload=dict(signal.payload),
                    )
                )
