import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from artlib_studio.core.events import Event, EventType
from artlib_studio.visualization.explanations import get_explanation

def test_explanations():
    ev = Event(EventType.INPUT_RECEIVED, timestamp=0, payload={"sample_index": 5})
    assert "received sample 5" in get_explanation(ev)

    ev2 = Event(EventType.CATEGORY_SELECTED, timestamp=0, payload={"category_id": 2})
    assert "Category 2" in get_explanation(ev2)

    ev3 = Event(EventType.MATCH_TEST, timestamp=0, payload={"passed": True})
    assert "(match >= vigilance)" in get_explanation(ev3)

from artlib_studio.visualization.competition_table import show_competition_table
from artlib_studio.visualization.search_history import show_search_history
from artlib_studio.visualization.match_vigilance import show_match_vigilance_panel
class DummyTrace:
    def __init__(self, events):
        self.events = events
def test_panels_handle_empty():
    # Should not crash
    show_competition_table(None, -1)
    show_search_history(None, -1)
    show_match_vigilance_panel(None)
    # Should handle empty trace
    trace = DummyTrace([])
    show_competition_table(trace, 0)
    show_search_history(trace, 0)
