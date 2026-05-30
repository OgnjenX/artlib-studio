"""Instrumentation and event recording for ARTLib Studio.

Provides Event, EventType and TraceRecorder used by the explorer.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from time import time
from typing import Any, Dict, List, Optional
import json


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


class TraceRecorder:
    """Simple in-memory trace recorder for ART events.

    Records events (append-only) and provides stepping utilities used by the
    explorer UI.
    """

    def __init__(self):
        self.events: List[Event] = []
        self.index: int = -1

    def record(self, etype: EventType, payload: Optional[Dict] = None):
        if payload is None:
            payload = {}
        ev = Event(type=etype, timestamp=time(), payload=payload)
        self.events.append(ev)
        self.index = len(self.events) - 1
        return ev

    def step_forward(self) -> Optional[Event]:
        if self.index + 1 < len(self.events):
            self.index += 1
            return self.events[self.index]
        return None

    def step_backward(self) -> Optional[Event]:
        if self.index - 1 >= 0:
            self.index -= 1
            return self.events[self.index]
        return None

    def current(self) -> Optional[Event]:
        if 0 <= self.index < len(self.events):
            return self.events[self.index]
        return None

    def replay(self):
        for ev in self.events:
            yield ev

    def to_json(self) -> str:
        return json.dumps([e.to_dict() for e in self.events], indent=2)

    def clear(self):
        self.events = []
        self.index = -1

