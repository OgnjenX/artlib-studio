"""Container and coordination API for composable ART modules."""
from __future__ import annotations

from typing import Dict, Iterable, List

from .edges import ModuleEdge
from .events import CompositionEvent
from .module import ComposableARTModule
from .scheduler import DiscreteScheduler
from .signals import CompositionSignal


class ARTCompositionGraph:
    def __init__(self, max_settling_steps: int = 5):
        self.modules: Dict[str, ComposableARTModule] = {}
        self.edges: List[ModuleEdge] = []
        self.scheduler = DiscreteScheduler(max_settling_steps=max_settling_steps)
        self._event_log: List[CompositionEvent] = []

    def add_module(self, module: ComposableARTModule) -> None:
        if module.module_id in self.modules:
            raise ValueError(f"Module {module.module_id!r} already exists")
        self.modules[module.module_id] = module

    def add_edge(self, edge: ModuleEdge) -> None:
        if edge.source_module_id not in self.modules:
            raise KeyError(f"Unknown source module {edge.source_module_id!r}")
        if edge.target_module_id not in self.modules:
            raise KeyError(f"Unknown target module {edge.target_module_id!r}")
        self.edges.append(edge)

    def get_module(self, module_id: str) -> ComposableARTModule:
        try:
            return self.modules[module_id]
        except KeyError as exc:
            raise KeyError(f"Unknown module {module_id!r}") from exc

    def outgoing_edges(self, module_id: str) -> List[ModuleEdge]:
        return [edge for edge in self.edges if edge.source_module_id == module_id]

    def step(self, input_signals: Iterable[CompositionSignal]) -> bool:
        return self.scheduler.run(self, input_signals)

    def get_global_state(self) -> Dict[str, object]:
        return {
            "step_index": self.scheduler.step_index,
            "modules": {
                module_id: module.get_state()
                for module_id, module in self.modules.items()
            },
            "edge_count": len(self.edges),
            "event_count": len(self._event_log),
        }

    def get_event_log(self) -> List[CompositionEvent]:
        return list(self._event_log)

    def _record(self, event: CompositionEvent) -> None:
        self._event_log.append(event)
