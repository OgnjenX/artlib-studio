"""An instrumented wrapper around ART modules from AdaptiveResonanceLib.

This wrapper delegates to an existing ART instance (e.g. FuzzyART) but emits
events to a TraceRecorder so the explorer UI can visualize internal execution
without modifying the original AdaptiveResonanceLib code.
"""
from typing import Optional, Callable, Any
import numpy as np
from artlib_studio.events import (
    TraceRecorder,
    EventType,
)


class InstrumentedART:
    """Wraps an ART instance and records execution events.

    The wrapper implements a step_fit surface compatible with BaseART.step_fit
    semantics but emits events during execution.
    """

    def __init__(self, art_instance: Any, recorder: Optional[TraceRecorder] = None):
        self.art = art_instance
        self.recorder = recorder or TraceRecorder()

    # convenience passthrough
    def __getattr__(self, key):
        return getattr(self.art, key)

    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        return self.art.prepare_data(X)

    def step_fit(
        self,
        x: np.ndarray,
        sample_index: int = -1,
        match_reset_func: Optional[Callable] = None,
        match_tracking: str = "MT+",
        epsilon: float = 0.0,
    ) -> int:
        """Instrumented single-sample fit.

        Mirrors BaseART.step_fit logic but records events. Returns the chosen
        cluster index.
        """
        # Record input
        self.recorder.record(
            EventType.INPUT_RECEIVED, 
            {
                "sample_index": sample_index,
                "input": x.tolist(),
                "explanation": f"Input received."
            }
        )

        # follow the same logic as BaseART.step_fit, but call through to the
        # wrapped art instance methods so behavior is unchanged.

        # increment sample counter on wrapped instance if present
        if hasattr(self.art, "sample_counter_"):
            self.art.sample_counter_ += 1

        base_params = self.art._deep_copy_params() if hasattr(self.art, "_deep_copy_params") else None

        # compute category choice for existing weights
        if not hasattr(self.art, "W") or len(self.art.W) == 0:
            # no categories -> create one
            w_new = self.art.new_weight(x, self.art.params)
            self.art.add_weight(w_new)
            self.recorder.record(
                EventType.CATEGORY_CREATED, 
                {
                    "sample_index": sample_index,
                    "created_index": 0,
                    "explanation": "No categories exist. New category 0 created.",
                }
            )
            return 0

        # compute T values
        if match_tracking == "MT~" and match_reset_func is not None:
            T_values = []
            T_cache = []
            for c_, w in enumerate(self.art.W):
                if match_reset_func(x, w, c_, params=self.art.params, cache=None):
                    t_val, cache = self.art.category_choice(x, w, params=self.art.params)
                else:
                    t_val, cache = (np.nan, None)
                T_values.append(t_val)
                T_cache.append(cache)
        else:
            T_values = []
            T_cache = []
            for w in self.art.W:
                t_val, cache = self.art.category_choice(x, w, params=self.art.params)
                T_values.append(t_val)
                T_cache.append(cache)

        T = np.array(T_values)

        # emit CATEGORY_EVALUATED events for transparency
        for idx, t in enumerate(T_values):
            ch_val = None if np.isnan(t) else float(t)
            self.recorder.record(
                EventType.CATEGORY_EVALUATED, 
                {
                    "sample_index": sample_index,
                    "category_id": int(idx), 
                    "choice_score": ch_val,
                    "explanation": f"Category {idx} evaluated. Choice score = {ch_val:.4f}" if ch_val is not None else f"Category {idx} evaluated. Choice score = NaN"
                }
            )

        # sort candidates
        valid = ~np.isnan(T)
        if np.any(valid):
            idxs = np.arange(T.shape[0])[valid]
            T_valid = T[valid]
            order = idxs[np.lexsort((idxs, -T_valid))]
        else:
            order = np.array([], dtype=int)

        # match operator from art
        mt_operator = self.art._match_tracking_operator(match_tracking)

        for c_ in order:
            w = self.art.W[c_]
            cache = T_cache[c_]

            # select candidate -> event
            self.recorder.record(
                EventType.CATEGORY_SELECTED, 
                {
                    "sample_index": sample_index,
                    "category_id": int(c_),
                    "explanation": f"Category {c_} selected as best candidate."
                }
            )

            m, cache = self.art.match_criterion_bin(
                x, w, params=self.art.params, cache=cache, op=mt_operator
            )

            # emit match test results
            mc = cache.get("match_criterion") if isinstance(cache, dict) else None
            vig = float(self.art.params.get("rho"))
            passed = bool(m)
            self.recorder.record(
                EventType.MATCH_TEST, 
                {
                    "sample_index": sample_index,
                    "category_id": int(c_),
                    "match_score": None if mc is None else float(mc),
                    "vigilance": vig,
                    "passed": passed,
                    "explanation": f"Match test: match = {float(mc):.4f}, vigilance = {vig:.4f}. {'Passed' if passed else 'Failed'}." if mc is not None else f"Match test: {'Passed' if passed else 'Failed'}."
                }
            )

            if match_tracking == "MT~" and match_reset_func is not None:
                no_match_reset = True
            else:
                no_match_reset = match_reset_func is None or match_reset_func(
                    x, w, c_, params=self.art.params, cache=cache
                )

            if m and no_match_reset:
                # Resonance
                self.recorder.record(
                    EventType.RESONANCE, 
                    {
                        "sample_index": sample_index,
                        "category_id": int(c_),
                        "explanation": f"Resonance occurred with category {c_}."
                    }
                )
                # Learning (update weight)
                new_w = self.art.update(x, w, self.art.params, cache=cache)
                self.art.set_weight(c_, new_w)
                self.recorder.record(
                    EventType.LEARNING, 
                    {
                        "sample_index": sample_index,
                        "category_id": int(c_),
                        "explanation": f"Category {c_} learned the input pattern."
                    }
                )
                # restore params
                if base_params is not None:
                    self.art._set_params(base_params)
                return c_
            else:
                if m and not no_match_reset:
                    # RESET event
                    self.recorder.record(
                        EventType.RESET, 
                        {
                            "sample_index": sample_index,
                            "category_id": int(c_),
                            "explanation": f"Reset category {c_}."
                        }
                    )
                    keep_searching = self.art._match_tracking(cache, epsilon, self.art.params, match_tracking)
                    if not keep_searching:
                        break

        # if none matched: create new category
        c_new = len(self.art.W)
        w_new = self.art.new_weight(x, self.art.params)
        self.art.add_weight(w_new)
        self.recorder.record(
            EventType.CATEGORY_CREATED, 
            {
                "sample_index": sample_index,
                "created_index": int(c_new),
                "explanation": f"Search continues. New category {c_new} created."
            }
        )
        if base_params is not None:
            self.art._set_params(base_params)
        return c_new

    def fit(self, X, **kwargs):
        """Convenience: run prepare_data if available and fit sample-by-sample using step_fit.

        Returns the wrapped art instance for chaining.
        """
        # If the underlying ART has prepare_data, call it
        try:
            Xp = self.art.prepare_data(X)
        except Exception:
            Xp = X

        if hasattr(self.art, "validate_data"):
            self.art.validate_data(Xp)
        if hasattr(self.art, "check_dimensions"):
            self.art.check_dimensions(Xp)

        self.art.is_fitted_ = True

        # Ensure the art instance is in a clean state similar to BaseART.fit
        self.art.W = []
        self.art.labels_ = np.zeros((Xp.shape[0],), dtype=int)

        for i, x in enumerate(Xp):
            self.art.labels_[i] = self.step_fit(x, sample_index=i, **kwargs)

        return self.art
