"""Typed signals exchanged by modules in an ART composition graph."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional


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
    pass


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
