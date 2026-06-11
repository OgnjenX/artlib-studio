"""Configuration schema and serialization for ART composition graphs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .edges import EdgeType


SUPPORTED_TRANSFORMS = {
    "selected_category_to_one_hot",
    "selected_category_to_scalar_vector",
    "selected_category_to_activation_vector",
    "high_category_to_expectation",
}
SUPPORTED_MODULE_TYPES = {"adapter", "context"}
SUPPORTED_ADAPTERS = {"fuzzy_art"}
SUPPORTED_MODULATION_MODES = {"set", "add", "multiply"}
SUPPORTED_MODULATION_DURATIONS = {"current_step", "persistent"}


@dataclass
class TransformConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextRuleConfig:
    name: str
    active: bool = True
    target_param: str = "rho"
    mode: str = "set"
    value: float = 0.85
    duration: str = "current_step"
    explanation: str = ""


@dataclass
class ModuleConfig:
    id: str
    type: str = "adapter"
    adapter: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    rules: List[ContextRuleConfig] = field(default_factory=list)


@dataclass
class EdgeConfig:
    source: str
    target: str
    type: str
    transform: Optional[TransformConfig] = None


@dataclass
class AssociationConfig:
    high_category_id: int
    low_category_id: int


@dataclass
class GraphConfig:
    name: str
    description: str = ""
    modules: List[ModuleConfig] = field(default_factory=list)
    edges: List[EdgeConfig] = field(default_factory=list)
    associations: List[AssociationConfig] = field(default_factory=list)
    max_settling_steps: int = 5

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Graph config requires a non-empty name")
        if self.max_settling_steps < 1:
            raise ValueError("max_settling_steps must be at least 1")

        module_ids = [module.id for module in self.modules]
        if len(module_ids) != len(set(module_ids)):
            raise ValueError("Module ids must be unique")
        known_modules = set(module_ids)

        for module in self.modules:
            if not module.id:
                raise ValueError("Module id must be non-empty")
            if module.type not in SUPPORTED_MODULE_TYPES:
                raise ValueError(
                    f"Unsupported module type {module.type!r} for {module.id!r}"
                )
            if module.type == "adapter":
                if module.adapter not in SUPPORTED_ADAPTERS:
                    raise ValueError(
                        f"Unsupported adapter {module.adapter!r} for {module.id!r}"
                    )
            if module.type == "context":
                if not module.rules:
                    raise ValueError(
                        f"Context module {module.id!r} requires at least one rule"
                    )
                for rule in module.rules:
                    _validate_context_rule(module.id, rule)

        for edge in self.edges:
            if edge.source not in known_modules:
                raise ValueError(
                    f"Edge source references unknown module {edge.source!r}"
                )
            if edge.target not in known_modules:
                raise ValueError(
                    f"Edge target references unknown module {edge.target!r}"
                )
            try:
                EdgeType(edge.type)
            except ValueError as exc:
                raise ValueError(f"Unsupported edge type {edge.type!r}") from exc
            if edge.transform is not None:
                _validate_transform(edge.transform)

        for association in self.associations:
            if not isinstance(association.high_category_id, int):
                raise ValueError("Association high_category_id must be an integer")
            if not isinstance(association.low_category_id, int):
                raise ValueError("Association low_category_id must be an integer")


def _validate_transform(transform: TransformConfig) -> None:
    if transform.name not in SUPPORTED_TRANSFORMS:
        raise ValueError(f"Unsupported transform {transform.name!r}")
    if transform.name == "selected_category_to_one_hot":
        vector_size = transform.params.get("vector_size")
        if not isinstance(vector_size, int) or vector_size < 1:
            raise ValueError(
                "selected_category_to_one_hot requires integer vector_size >= 1"
            )
    if transform.name == "selected_category_to_scalar_vector":
        maximum = transform.params.get("max_category_id")
        if maximum is not None and (
            not isinstance(maximum, int) or maximum < 1
        ):
            raise ValueError("max_category_id must be an integer >= 1")


def _validate_context_rule(module_id: str, rule: ContextRuleConfig) -> None:
    if rule.target_param != "rho":
        raise ValueError(
            f"Context module {module_id!r} only supports target_param 'rho'"
        )
    if rule.mode not in SUPPORTED_MODULATION_MODES:
        raise ValueError(f"Unsupported modulation mode {rule.mode!r}")
    if rule.duration not in SUPPORTED_MODULATION_DURATIONS:
        raise ValueError(f"Unsupported modulation duration {rule.duration!r}")


def graph_config_from_dict(data: Dict[str, Any]) -> GraphConfig:
    modules = []
    for module_data in data.get("modules", []):
        rules = [
            ContextRuleConfig(**rule)
            for rule in module_data.get("rules", [])
        ]
        modules.append(
            ModuleConfig(
                id=module_data["id"],
                type=module_data.get("type", "adapter"),
                adapter=module_data.get("adapter"),
                params=dict(module_data.get("params", {})),
                rules=rules,
            )
        )
    edges = []
    for edge_data in data.get("edges", []):
        transform_data = edge_data.get("transform")
        transform = (
            TransformConfig(
                name=transform_data["name"],
                params=dict(transform_data.get("params", {})),
            )
            if transform_data
            else None
        )
        edges.append(
            EdgeConfig(
                source=edge_data["source"],
                target=edge_data["target"],
                type=edge_data["type"],
                transform=transform,
            )
        )
    associations = [
        AssociationConfig(**association)
        for association in data.get("associations", [])
    ]
    config = GraphConfig(
        name=data.get("name", ""),
        description=data.get("description", ""),
        modules=modules,
        edges=edges,
        associations=associations,
        max_settling_steps=int(data.get("max_settling_steps", 5)),
    )
    config.validate()
    return config


def graph_config_to_dict(config: GraphConfig) -> Dict[str, Any]:
    config.validate()
    return asdict(config)


def load_graph_config(path: Union[str, Path]) -> GraphConfig:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported config format {suffix!r}")
    if not isinstance(data, dict):
        raise ValueError("Graph config root must be an object")
    return graph_config_from_dict(data)


def save_graph_config(config: GraphConfig, path: Union[str, Path]) -> Path:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = graph_config_to_dict(config)
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        text = json.dumps(data, indent=2)
    elif suffix in {".yaml", ".yml"}:
        text = yaml.safe_dump(data, sort_keys=False)
    else:
        raise ValueError(f"Unsupported config format {suffix!r}")
    config_path.write_text(text, encoding="utf-8")
    return config_path
