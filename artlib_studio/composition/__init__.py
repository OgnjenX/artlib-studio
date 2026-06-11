"""Composable ART module protocol and graph runtime."""

from .association_memory import AssociationMemory
from .builder import build_graph_from_config
from .config import (
    AssociationConfig,
    ContextRuleConfig,
    EdgeConfig,
    GraphConfig,
    ModuleConfig,
    TransformConfig,
    graph_config_from_dict,
    graph_config_to_dict,
    load_graph_config,
    save_graph_config,
)
from .context import ContextModule, ContextRule
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
    ModulatorySignal,
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
    "AssociationConfig",
    "AdapterARTModule",
    "CategoryActivationSignal",
    "ComposableARTModule",
    "CompositionEvent",
    "CompositionEventType",
    "ContextRuleConfig",
    "ContextModule",
    "ContextRule",
    "EdgeConfig",
    "EdgeType",
    "ExpectationSignal",
    "GraphConfig",
    "InputSignal",
    "LearningSignal",
    "MatchSignal",
    "ModuleEdge",
    "ModuleConfig",
    "ModulatorySignal",
    "ResetSignal",
    "ResonanceSignal",
    "SelectedCategorySignal",
    "TransformConfig",
    "build_graph_from_config",
    "graph_config_from_dict",
    "graph_config_to_dict",
    "high_category_to_expectation",
    "selected_category_to_activation_vector",
    "selected_category_to_one_hot",
    "selected_category_to_scalar_vector",
    "load_graph_config",
    "save_graph_config",
]
