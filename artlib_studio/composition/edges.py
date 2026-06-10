"""Edges connecting modules in an ART composition graph."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from .signals import CompositionSignal


class EdgeType(str, Enum):
    BOTTOM_UP = "BOTTOM_UP"
    TOP_DOWN_EXPECTATION = "TOP_DOWN_EXPECTATION"
    MODULATORY = "MODULATORY"
    ASSOCIATIVE = "ASSOCIATIVE"
    RESET_PROPAGATION = "RESET_PROPAGATION"


SignalTransform = Callable[[CompositionSignal], CompositionSignal]


@dataclass(frozen=True)
class ModuleEdge:
    source_module_id: str
    target_module_id: str
    edge_type: EdgeType
    transform: Optional[SignalTransform] = None
    transform_name: Optional[str] = None

    def transmit(self, signal: CompositionSignal) -> CompositionSignal:
        transformed = self.transform(signal) if self.transform else signal
        return transformed.for_target(self.target_module_id)
