"""Build executable ART composition graphs from validated configuration."""
from __future__ import annotations

from functools import partial
from typing import Callable, Optional

from ..core.registry import get_adapter
from .config import GraphConfig, TransformConfig
from .context import ContextModule, ContextRule
from .edges import EdgeType, ModuleEdge, SignalTransform
from .graph import ARTCompositionGraph
from .module import AdapterARTModule
from .transforms import (
    high_category_to_expectation,
    selected_category_to_activation_vector,
    selected_category_to_one_hot,
    selected_category_to_scalar_vector,
)


def build_graph_from_config(config: GraphConfig) -> ARTCompositionGraph:
    config.validate()
    graph = ARTCompositionGraph(
        max_settling_steps=config.max_settling_steps
    )
    for module_config in config.modules:
        if module_config.type == "adapter":
            graph.add_module(
                AdapterARTModule(
                    module_config.id,
                    get_adapter(module_config.adapter or ""),
                    module_config.params,
                )
            )
        elif module_config.type == "context":
            graph.add_module(
                ContextModule(
                    module_config.id,
                    rules=[
                        ContextRule(
                            name=rule.name,
                            active=rule.active,
                            target_param=rule.target_param,
                            mode=rule.mode,
                            value=rule.value,
                            duration=rule.duration,
                            explanation=rule.explanation,
                        )
                        for rule in module_config.rules
                    ],
                )
            )
    for association in config.associations:
        graph.association_memory.record_pair(
            association.high_category_id,
            association.low_category_id,
        )
    for edge_config in config.edges:
        graph.add_edge(
            ModuleEdge(
                source_module_id=edge_config.source,
                target_module_id=edge_config.target,
                edge_type=EdgeType(edge_config.type),
                transform=_resolve_transform(graph, edge_config.transform),
                transform_name=(
                    edge_config.transform.name
                    if edge_config.transform is not None
                    else None
                ),
            )
        )
    return graph


def _resolve_transform(
    graph: ARTCompositionGraph,
    config: Optional[TransformConfig],
) -> Optional[SignalTransform]:
    if config is None:
        return None
    params = dict(config.params)
    transforms: dict[str, Callable] = {
        "selected_category_to_one_hot": selected_category_to_one_hot,
        "selected_category_to_scalar_vector": selected_category_to_scalar_vector,
        "selected_category_to_activation_vector": (
            selected_category_to_activation_vector
        ),
    }
    if config.name == "high_category_to_expectation":
        return partial(
            high_category_to_expectation,
            association_memory=graph.association_memory,
        )
    transform = transforms.get(config.name)
    if transform is None:
        raise ValueError(f"Unsupported transform {config.name!r}")
    return partial(transform, **params) if params else transform
