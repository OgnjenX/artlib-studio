"""Composable ART module protocol and graph runtime."""

from .association_memory import AssociationMemory
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
from .transforms import (
    high_category_to_expectation,
    selected_category_to_activation_vector,
    selected_category_to_one_hot,
    selected_category_to_scalar_vector,
)

__all__ = [
    "ARTCompositionGraph",
    "AssociationMemory",
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
    "high_category_to_expectation",
    "selected_category_to_activation_vector",
    "selected_category_to_one_hot",
    "selected_category_to_scalar_vector",
]
