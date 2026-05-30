"""Gaussian ART adapter for ARTLib Studio."""
from typing import Any, Dict, List, Optional, Set
import numpy as np
from artlib.elementary.GaussianART import GaussianART
from ..core.model_adapter import ARTModelAdapter
from ..core.capabilities import Capability
from ..core.recorder import TraceRecorder
from ..instrumentation.instrumented_art import InstrumentedART


class GaussianARTAdapter(ARTModelAdapter):
    @property
    def name(self) -> str:
        return "Gaussian ART"

    @property
    def model_key(self) -> str:
        return "gaussian_art"

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
            Capability.GAUSSIAN_REGIONS_2D,
            Capability.CATEGORY_PROTOTYPES,
            Capability.STREAMLIT_EXPLORER
        }

    def create_model(self, params: dict) -> Any:
        # We need dim_ to build sigma_init.
        # Usually GaussianART has sigma_init array.
        # We'll expect params to pass "sigma_init" array, or we'll construct it in fit_with_trace
        sigma_val = float(params.get("sigma_init_scalar", 0.1))
        # Default fallback if array not passed
        sigma_arr = params.get("sigma_init", np.array([sigma_val, sigma_val]))
        return StudioGaussianART(
            rho=float(params.get("rho", 0.5)),
            alpha=float(params.get("alpha", 1e-10)),
            sigma_init=sigma_arr
        )

    def default_params(self) -> dict:
        return {
            "rho": 0.5,
            "alpha": 1e-10,
            "sigma_init_scalar": 0.1
        }

    def param_schema(self) -> dict:
        return {
            "rho": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5, "step": 0.01, "label": "vigilance (rho)"},
            "sigma_init_scalar": {"type": "float", "min": 0.001, "max": 1.0, "default": 0.1, "step": 0.01, "label": "initial std dev (sigma_init)"},
            "alpha": {"type": "float", "min": 1e-12, "max": 1.0, "default": 1e-10, "format": "%.1e", "label": "choice threshold (alpha)"},
        }

    def fit_with_trace(self, X: Any, params: dict, recorder: Optional[TraceRecorder] = None) -> Any:
        # Convert scalar to array based on X dimensions
        sigma_val = params.get("sigma_init_scalar", 0.1)
        dim = X.shape[1] if hasattr(X, "shape") else len(X[0])
        params["sigma_init"] = np.array([sigma_val] * dim)

        model = self.create_model(params)
        instr = InstrumentedART(model, recorder=recorder)
        instr.fit(X)
        return instr

    def predict(self, model: Any, X: Any) -> Any:
        return model.predict(X)

    def get_category_geometry_2d(self, model: Any) -> Any:
        """Return gaussian parameters (mean, sigma)."""
        gaussians = []
        if hasattr(model, "W") and hasattr(model, "dim_"):
            for idx, w in enumerate(model.W):
                # GaussianART weight: [mean_1..mean_d, sigma_1..sigma_d, inv_sigma..., det_sigma, n]
                mean = w[: model.dim_]
                sigma = w[model.dim_ : 2 * model.dim_]
                if len(mean) >= 2:
                    gaussians.append({
                        "id": idx,
                        "x": mean[0],
                        "y": mean[1],
                        "sigma_x": sigma[0],
                        "sigma_y": sigma[1]
                    })
        return gaussians

    def get_category_prototypes(self, model: Any) -> Any:
        if hasattr(model, "W"):
            return [w.tolist() for w in model.W]
        return []

    def build_competition_table(self, trace_events: List[Any], sample_index: int) -> Any:
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
                "Choice score": f"{score:.4e}" if score is not None else "NaN",
                "Match score": f"{m_score:.4e}" if m_score is not None else "—",
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


class StudioGaussianART(GaussianART):
    """GaussianART with NumPy 2.x compatible category choice."""

    def category_choice(self, i: np.ndarray, w: np.ndarray, params: dict) -> tuple[float, Optional[dict]]:
        mean = w[: self.dim_]
        inv_sig = w[2 * self.dim_ : 3 * self.dim_]
        sqrt_det_sig = w[-2]
        n = w[-1]

        dist = mean - i
        exp_dist_sig_dist = np.exp(-0.5 * np.dot(dist, np.multiply(inv_sig, dist)))

        cache = {"exp_dist_sig_dist": exp_dist_sig_dist}
        p_i_cj = exp_dist_sig_dist / (params["alpha"] + sqrt_det_sig)
        p_cj = n / sum(w_[-1] for w_ in self.W)

        return p_i_cj * p_cj, cache
