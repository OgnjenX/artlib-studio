"""Event definitions for ARTLib Studio."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict


class EventType(str, Enum):
    INPUT_RECEIVED = "INPUT_RECEIVED"
    CATEGORY_EVALUATED = "CATEGORY_EVALUATED"
    CATEGORY_SELECTED = "CATEGORY_SELECTED"
    MATCH_TEST = "MATCH_TEST"
    RESET = "RESET"
    RESONANCE = "RESONANCE"
    LEARNING = "LEARNING"
    CATEGORY_CREATED = "CATEGORY_CREATED"


@dataclass
class Event:
    type: EventType
    timestamp: float
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # convert enum to value for JSON friendliness
        d["type"] = self.type.value
        return d

