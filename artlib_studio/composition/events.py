"""Events produced by the composition runtime."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from time import time
from typing import Any, Dict, Optional


class CompositionEventType(str, Enum):
    MODULE_RECEIVED_INPUT = "MODULE_RECEIVED_INPUT"
    MODULE_CATEGORY_CREATED = "MODULE_CATEGORY_CREATED"
    MODULE_SELECTED_CATEGORY = "MODULE_SELECTED_CATEGORY"
    MODULE_RESONATED = "MODULE_RESONATED"
    MODULE_RESET = "MODULE_RESET"
    MODULE_LEARNED = "MODULE_LEARNED"
    SIGNAL_SENT = "SIGNAL_SENT"
    SIGNAL_RECEIVED = "SIGNAL_RECEIVED"
    EXPECTATION_SENT = "EXPECTATION_SENT"
    EXPECTATION_RECEIVED = "EXPECTATION_RECEIVED"
    EXPECTATION_UNAVAILABLE = "EXPECTATION_UNAVAILABLE"
    EXPECTATION_MATCHED = "EXPECTATION_MATCHED"
    EXPECTATION_MISMATCHED = "EXPECTATION_MISMATCHED"
    CROSS_MODULE_EXPECTATION_UNKNOWN = "CROSS_MODULE_EXPECTATION_UNKNOWN"
    CROSS_MODULE_RESONANCE = "CROSS_MODULE_RESONANCE"
    CROSS_MODULE_MISMATCH = "CROSS_MODULE_MISMATCH"
    GRAPH_SETTLED = "GRAPH_SETTLED"
    GRAPH_FAILED_TO_SETTLE = "GRAPH_FAILED_TO_SETTLE"


@dataclass(frozen=True)
class CompositionEvent:
    type: CompositionEventType
    step_index: int
    module_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value
        data["external_sample_index"] = self.step_index
        return data
