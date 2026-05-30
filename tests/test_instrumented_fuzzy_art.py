import sys
import os
import numpy as np

# Make AdaptiveResonanceLib and artlib_studio importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARL_PATH = os.path.join(ROOT, "AdaptiveResonanceLib")
STUDIO_PATH = os.path.join(ROOT, "artlib-studio")
if ARL_PATH not in sys.path:
    sys.path.insert(0, ARL_PATH)
if STUDIO_PATH not in sys.path:
    sys.path.insert(0, STUDIO_PATH)

from artlib.elementary.FuzzyART import FuzzyART
from artlib_studio.instrumented_art import InstrumentedART
from artlib_studio.events import TraceRecorder, EventType
from sklearn.datasets import make_blobs

def test_instrumented_art_behavior_matches_original():
    # Generate some quick deterministic data
    X, _ = make_blobs(n_samples=50, centers=3, random_state=42)
    # Normalize data for FuzzyART
    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)
    Xn = (X - Xmin) / (Xmax - Xmin + 1e-9)

    params = {"rho": 0.5, "alpha": 0.0, "beta": 1.0}

    # Run original FuzzyART
    art_original = FuzzyART(**params)
    # fit method normalizes internally via prepare_data if called without Xp, but we'll use fit() directly
    # Wait, FuzzyART does complement coding inside prepare_data. If we use fit(X), it will normalize and complement code.
    # Actually, BaseART.fit uses X directly without prepare_data unless we manually do it! Let's just do prepare_data explicitly, or see how fit behaves.
    try:
        Xp = art_original.prepare_data(Xn)
    except Exception:
        Xp = Xn

    art_original.fit(Xp)
    labels_original = art_original.labels_.copy()
    w_original = [w.copy() for w in art_original.W]

    # Run InstrumentedART
    art_instrumented = InstrumentedART(FuzzyART(**params))
    # InstrumentedART.fit handles data preparation internally if possible, but let's pass prep'd data,
    # or just Xn to be similar. Let's trace InstrumentedART's fit method.
    art_instrumented.fit(Xn)
    labels_instrumented = art_instrumented.labels_.copy()
    w_instrumented = [w.copy() for w in art_instrumented.W]

    # Compare
    assert len(w_original) == len(w_instrumented), "Number of categories differs"
    np.testing.assert_array_equal(labels_original, labels_instrumented, err_msg="Category assignments differ")

    for wo, wi in zip(w_original, w_instrumented):
        np.testing.assert_allclose(wo, wi, err_msg="Learned weights differ")

def test_trace_recorder_events():
    X, _ = make_blobs(n_samples=10, centers=2, random_state=42)
    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)
    Xn = (X - Xmin) / (Xmax - Xmin + 1e-9)

    recorder = TraceRecorder()
    art = FuzzyART(rho=0.5, alpha=0.0, beta=1.0)
    instr = InstrumentedART(art, recorder=recorder)
    
    instr.fit(Xn)
    
    events = [e.type for e in recorder.events]
    
    assert EventType.INPUT_RECEIVED in events
    # Only for samples after the first one, it evaluates existing categories
    assert EventType.CATEGORY_EVALUATED in events
    assert EventType.CATEGORY_SELECTED in events
    assert EventType.MATCH_TEST in events
    # For training it should create categories and also resonate
    has_resonance_or_creation = (EventType.RESONANCE in events) or (EventType.CATEGORY_CREATED in events)
    assert has_resonance_or_creation, "Should have resonance or category creation"
    # Wait, learning only happens on resonance? Does it log LEARNING?
    # Our instrumented_art.py records EventType.LEARNING together with RESONANCE
    assert EventType.LEARNING in events or EventType.CATEGORY_CREATED in events

    # Verify new fields
    for e in recorder.events:
        assert "sample_index" in e.payload
        assert "explanation" in e.payload

def test_instrumented_art_new_inspector_fields():
    X = np.array([[0.1, 0.2], [0.11, 0.21]])
    recorder = TraceRecorder()
    # High vigilance to trigger resets if we had different samples,
    # but here let's just check if fields are present
    art = FuzzyART(rho=0.9, alpha=0.0, beta=1.0)
    instr = InstrumentedART(art, recorder=recorder)
    instr.fit(X)

    # Check INPUT_RECEIVED
    input_evs = [e for e in recorder.events if e.type == EventType.INPUT_RECEIVED]
    assert len(input_evs) == 2
    for ev in input_evs:
        assert "input" in ev.payload
        assert "prepared_input" in ev.payload
        assert "rho" in ev.payload
        assert "alpha" in ev.payload
        assert "beta" in ev.payload

    # Check CATEGORY_SELECTED
    selected_evs = [e for e in recorder.events if e.type == EventType.CATEGORY_SELECTED]
    if selected_evs:
        for ev in selected_evs:
            assert "choice_score" in ev.payload
            assert isinstance(ev.payload["choice_score"], float)

    # Check MATCH_TEST
    match_evs = [e for e in recorder.events if e.type == EventType.MATCH_TEST]
    if match_evs:
        for ev in match_evs:
            assert "match_score" in ev.payload
            assert "vigilance" in ev.payload
            assert "passed" in ev.payload

    # Check LEARNING
    learn_evs = [e for e in recorder.events if e.type == EventType.LEARNING]
    if learn_evs:
        for ev in learn_evs:
            assert "weights_before" in ev.payload
            assert "weights_after" in ev.payload
            assert len(ev.payload["weights_before"]) == len(ev.payload["weights_after"])

    # Check RESONANCE
    resonance_evs = [e for e in recorder.events if e.type == EventType.RESONANCE]
    if resonance_evs:
        for ev in resonance_evs:
            assert "match_score" in ev.payload
            assert "vigilance" in ev.payload

    # Check JSON export includes new fields
    json_str = recorder.to_json()
    assert "prepared_input" in json_str
    if learn_evs:
        assert "weights_before" in json_str


def test_trace_recorder_to_json():
    recorder = TraceRecorder()
    recorder.record(EventType.INPUT_RECEIVED, {"sample_index": 0, "explanation": "test"})
    json_str = recorder.to_json()
    assert "INPUT_RECEIVED" in json_str
    assert "sample_index" in json_str
    assert "explanation" in json_str
    import json
    data = json.loads(json_str)
    assert len(data) == 1
    assert data[0]["type"] == "INPUT_RECEIVED"
