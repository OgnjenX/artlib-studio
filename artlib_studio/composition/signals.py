"""Typed signals exchanged by modules in an ART composition graph."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CompositionSignal:
    source_module_id: str
    target_module_id: Optional[str] = None
    step_index: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def for_target(self, target_module_id: str) -> "CompositionSignal":
        """Return a copy addressed to a graph module."""
        return replace(self, target_module_id=target_module_id)


@dataclass(frozen=True)
class InputSignal(CompositionSignal):
    pass


@dataclass(frozen=True)
class CategoryActivationSignal(CompositionSignal):
    pass


@dataclass(frozen=True)
class SelectedCategorySignal(CompositionSignal):
    pass


@dataclass(frozen=True)
class ExpectationSignal(CompositionSignal):
    expected_category_id: Optional[int] = None
    expected_activation_vector: Optional[List[float]] = None
    confidence: Optional[float] = None
    explanation: str = ""

    @property
    def expected_category_ids(self) -> List[int]:
        values = self.payload.get("expected_category_ids")
        if values is not None:
            return [int(value) for value in values]
        if self.expected_category_id is not None:
            return [int(self.expected_category_id)]
        return []


@dataclass(frozen=True)
class ModulatorySignal(CompositionSignal):
    target_param: str = "rho"
    mode: str = "set"
    value: float = 0.0
    duration: str = "current_step"
    explanation: str = ""


@dataclass(frozen=True)
class MatchSignal(CompositionSignal):
    pass


@dataclass(frozen=True)
class ResetSignal(CompositionSignal):
    pass


@dataclass(frozen=True)
class ResonanceSignal(CompositionSignal):
    pass


@dataclass(frozen=True)
class LearningSignal(CompositionSignal):
    pass
