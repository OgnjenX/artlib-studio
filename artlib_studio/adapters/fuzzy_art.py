"""Fuzzy ART adapter for ARTLib Studio."""
from typing import Any, Dict, List, Optional, Set
import numpy as np
from artlib.elementary.FuzzyART import FuzzyART
from ..core.model_adapter import ARTModelAdapter
from ..core.capabilities import Capability
from ..core.recorder import TraceRecorder
from ..instrumentation.instrumented_art import InstrumentedART


class FuzzyARTAdapter(ARTModelAdapter):
    @property
    def name(self) -> str:
        return "Fuzzy ART"

    @property
    def model_key(self) -> str:
        return "fuzzy_art"

    @property
    def capabilities(self) -> Set[Capability]:
        return {
            Capability.TRACE_EXECUTION,
            Capability.CHOICE_SCORES,
            Capability.MATCH_SCORES,
            Capability.VIGILANCE,
            Capability.RESET_SEARCH,
            Capability.LEARNING_UPDATES,
            Capability.WEIGHT_BEFORE_AFTER,
            Capability.COMPLEMENT_CODING,
            Capability.CATEGORY_BOXES_2D,
            Capability.CATEGORY_PROTOTYPES,
            Capability.STREAMLIT_EXPLORER
        }

    def create_model(self, params: dict) -> Any:
        return FuzzyART(
            rho=float(params.get("rho", 0.85)),
            alpha=float(params.get("alpha", 0.0)),
            beta=float(params.get("beta", 1.0))
        )

    def default_params(self) -> dict:
        return {
            "rho": 0.85,
            "alpha": 0.0,
            "beta": 1.0
        }

    def param_schema(self) -> dict:
        return {
            "rho": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.85, "step": 0.01, "label": "vigilance (rho)"},
            "alpha": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0, "step": 0.01, "label": "choice (alpha)"},
            "beta": {"type": "float", "min": 0.0, "max": 1.0, "default": 1.0, "step": 0.01, "label": "learning rate (beta)"},
        }

    def prepare_data(self, X: Any) -> Any:
        # Just use the model's prepare_data directly from a fresh model to avoid state manipulation if you want to
        temp_model = self.create_model(self.default_params())
        return temp_model.prepare_data(X)

    def fit_with_trace(self, X: Any, params: dict, recorder: Optional[TraceRecorder] = None) -> Any:
        model = self.create_model(params)
        instr = InstrumentedART(model, recorder=recorder)
        instr.fit(X)
        return instr

    def predict(self, model: Any, X: Any) -> Any:
        return model.predict(X)

    def get_category_geometry_2d(self, model: Any) -> Any:
        """Return bounding boxes if the model has them."""
        bboxes = []
        if hasattr(model, "get_bounding_boxes") and getattr(model, "n_clusters", 0) > 0:
            boxes = model.get_bounding_boxes(n=2)
            for idx, bbox in enumerate(boxes):
                ref, widths = bbox
                bboxes.append({
                    "id": idx,
                    "x": ref[0],
                    "y": ref[1],
                    "width": widths[0],
                    "height": widths[1]
                })
        return bboxes

    def get_category_prototypes(self, model: Any) -> Any:
        """Return the actual learned weights/prototypes."""
        if hasattr(model, "W"):
            return [w.tolist() for w in model.W]
        return []

    def build_competition_table(self, trace_events: List[Any], sample_index: int) -> Any:
        """Build data for choice/match visualization using trace events for a sample."""
        # Logic is similar to what's in streamlit_app.py
        # Extract events for this sample
        sample_events = [e for e in trace_events if e.payload.get("sample_index") == sample_index]
        eval_events = [e for e in sample_events if e.type.value == "CATEGORY_EVALUATED"]
        match_events = [e for e in sample_events if e.type.value == "MATCH_TEST"]
        reset_events = [e for e in sample_events if e.type.value == "RESET"]
        resonance_ev = next((e for e in sample_events if e.type.value == "RESONANCE"), None)
        created_ev = next((e for e in sample_events if e.type.value == "CATEGORY_CREATED"), None)
        selected_evs = [e for e in sample_events if e.type.value == "CATEGORY_SELECTED"]
        last_selected_id = selected_evs[-1].payload.get("category_id") if selected_evs else None

        comp_data = []
        for ev in eval_events:
            cid = ev.payload.get("category_id")
            score = ev.payload.get("choice_score")

            m_ev = next((e for e in match_events if e.payload.get("category_id") == cid), None)
            m_score = m_ev.payload.get("match_score") if m_ev else None
            vig = m_ev.payload.get("vigilance") if m_ev else None

            status = "not evaluated"
            if m_ev:
                if any(r.payload.get("category_id") == cid for r in reset_events):
                    status = "reset"
                elif resonance_ev and resonance_ev.payload.get("category_id") == cid:
                    status = "resonance"
                else:
                    status = "selected"
            elif last_selected_id is not None and cid == last_selected_id:
                status = "selected"
            elif cid is not None:
                status = "not selected"

            comp_data.append({
                "Category": cid,
                "Choice score": f"{score:.4f}" if score is not None else "NaN",
                "Match score": f"{m_score:.4f}" if m_score is not None else "—",
                "Vigilance": f"{vig:.4f}" if vig is not None else "—",
                "Result": status
            })

        if created_ev:
            comp_data.append({
                "Category": created_ev.payload.get("created_index"),
                "Choice score": "—",
                "Match score": "—",
                "Vigilance": "—",
                "Result": "new category"
            })

        return comp_data

    def explain_event(self, event: Any) -> str:
        return event.payload.get("explanation", event.type.value)

