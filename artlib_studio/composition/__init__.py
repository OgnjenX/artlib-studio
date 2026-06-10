"""Composable ART module protocol and graph runtime."""

from .edges import EdgeType, ModuleEdge
from .events import CompositionEvent, CompositionEventType
from .graph import ARTCompositionGraph
from .module import AdapterARTModule, ComposableARTModule
from .signals import (
    CategoryActivationSignal,
    ExpectationSignal,
    InputSignal,
    LearningSignal,
    MatchSignal,
    ResetSignal,
    ResonanceSignal,
    SelectedCategorySignal,
)

__all__ = [
    "ARTCompositionGraph",
    "AdapterARTModule",
    "CategoryActivationSignal",
    "ComposableARTModule",
    "CompositionEvent",
    "CompositionEventType",
    "EdgeType",
    "ExpectationSignal",
    "InputSignal",
    "LearningSignal",
    "MatchSignal",
    "ModuleEdge",
    "ResetSignal",
    "ResonanceSignal",
    "SelectedCategorySignal",
]
