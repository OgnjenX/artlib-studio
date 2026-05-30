import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from artlib_studio.core.events import Event, EventType
from artlib_studio.visualization.timeline_state import build_timeline_state

def test_timeline_reconstruction():
    events = [
        Event(EventType.INPUT_RECEIVED, timestamp=0, payload={"sample_index": 0}),
        Event(EventType.CATEGORY_EVALUATED, timestamp=1, payload={"sample_index": 0, "category_id": 0}),
        Event(EventType.CATEGORY_CREATED, timestamp=2, payload={"sample_index": 0, "created_index": 0, "weights_after": [0.5, 0.5]}),
        Event(EventType.INPUT_RECEIVED, timestamp=3, payload={"sample_index": 1}),
        Event(EventType.CATEGORY_EVALUATED, timestamp=4, payload={"sample_index": 1, "category_id": 0}),
        Event(EventType.RESET, timestamp=5, payload={"sample_index": 1, "category_id": 0, "reset_categories": [0]}),
        Event(EventType.CATEGORY_CREATED, timestamp=6, payload={"sample_index": 1, "created_index": 1, "weights_after": [0.8, 0.8]})
    ]

    class DummyTrace:
        def __init__(self, evs):
            self.events = evs

    trace = DummyTrace(events)
    X = [[0,0], [1,1], [2,2], [3,3]]

    state = build_timeline_state(X, trace, 0)
    assert len(state["processed_sample_indices"]) == 1
    assert 0 in state["processed_sample_indices"]
    assert 1 not in state["processed_sample_indices"]
    assert len(state["categories_existing_so_far"]) == 0
    assert state["current_sample_index"] == 0

    state = build_timeline_state(X, trace, 2)
    assert 0 in state["categories_existing_so_far"]
    assert state["historical_weights"][0] == [0.5, 0.5]

    state = build_timeline_state(X, trace, 3)
    assert len(state["processed_sample_indices"]) == 2
    assert 1 in state["processed_sample_indices"]
    assert 1 not in state["categories_existing_so_far"]

    state = build_timeline_state(X, trace, 5)
    assert 0 in state["reset_categories_for_current_sample"]
    assert state["current_category_id"] == 0

