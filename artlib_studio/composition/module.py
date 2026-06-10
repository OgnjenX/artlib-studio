"""Protocol and adapter-backed implementation for composable ART modules."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.capabilities import Capability
from ..core.events import EventType
from ..core.model_adapter import ARTModelAdapter
from ..core.recorder import TraceRecorder
from ..instrumentation.instrumented_art import InstrumentedART
from .signals import (
    CategoryActivationSignal,
    CompositionSignal,
    InputSignal,
    LearningSignal,
    MatchSignal,
    ResetSignal,
    ResonanceSignal,
    SelectedCategorySignal,
)


class ComposableARTModule(ABC):
    """Minimal computational contract for a module in a composition graph."""

    def __init__(self, module_id: str):
        if not module_id:
            raise ValueError("module_id must be non-empty")
        self.module_id = module_id

    @abstractmethod
    def receive(self, signal: CompositionSignal) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_output_signals(self) -> List[CompositionSignal]:
        pass

    @abstractmethod
    def reset_runtime_state(self) -> None:
        pass

    @abstractmethod
    def supports_expectation(self) -> bool:
        pass

    @abstractmethod
    def explain_state(self) -> str:
        pass


class AdapterARTModule(ComposableARTModule):
    """Incremental composition wrapper for a BaseART-style Studio adapter."""

    def __init__(
        self,
        module_id: str,
        adapter: ARTModelAdapter,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(module_id)
        self.adapter = adapter
        self.model_params = {**adapter.default_params(), **(model_params or {})}
        self.trace_recorder = TraceRecorder()
        self._inbox: List[CompositionSignal] = []
        self._output_signals: List[CompositionSignal] = []
        self._sample_index = 0
        self._selected_category: Optional[int] = None
        self._last_trace_events = []
        self._build_runtime()

    def _build_runtime(self) -> None:
        model = self.adapter.create_model(self.model_params)
        self.model = InstrumentedART(model, recorder=self.trace_recorder)
        self.model.art.W = []
        self.model.art.is_fitted_ = True

    def receive(self, signal: CompositionSignal) -> None:
        self._inbox.append(signal)

    def step(self) -> None:
        self._output_signals = []
        pending = self._inbox
        self._inbox = []
        for signal in pending:
            if isinstance(signal, InputSignal):
                self._process_input(signal)

    def _process_input(self, signal: InputSignal) -> None:
        value = signal.payload.get("input", signal.payload.get("value"))
        if value is None:
            raise ValueError("InputSignal payload must contain 'input' or 'value'")

        raw = np.asarray(value, dtype=float)
        if self.adapter.supports(Capability.COMPLEMENT_CODING):
            if np.any(raw < 0.0) or np.any(raw > 1.0):
                raise ValueError(
                    "Streaming complement-coded input must already be in [0, 1]"
                )
            prepared = np.concatenate((raw, 1.0 - raw))
        else:
            prepared = self.adapter.prepare_data(np.atleast_2d(raw))[0]
        event_start = len(self.trace_recorder.events)
        selected = self.model.step_fit(
            prepared,
            sample_index=self._sample_index,
            raw_x=raw,
        )
        self._selected_category = int(selected)
        self._last_trace_events = self.trace_recorder.events[event_start:]
        self._output_signals = self._signals_from_trace(signal.step_index)
        self._sample_index += 1

    def _signals_from_trace(self, step_index: int) -> List[CompositionSignal]:
        outputs: List[CompositionSignal] = []
        for event in self._last_trace_events:
            payload = dict(event.payload)
            summary = payload.get("explanation", event.type.value)
            common = {
                "source_module_id": self.module_id,
                "step_index": step_index,
                "payload": payload,
                "summary": summary,
            }
            if event.type == EventType.CATEGORY_EVALUATED:
                outputs.append(CategoryActivationSignal(**common))
            elif event.type == EventType.CATEGORY_SELECTED:
                outputs.append(SelectedCategorySignal(**common))
            elif event.type == EventType.MATCH_TEST:
                outputs.append(MatchSignal(**common))
            elif event.type == EventType.RESET:
                outputs.append(ResetSignal(**common))
            elif event.type == EventType.RESONANCE:
                outputs.append(ResonanceSignal(**common))
            elif event.type == EventType.LEARNING:
                outputs.append(LearningSignal(**common))

        if not any(isinstance(s, SelectedCategorySignal) for s in outputs):
            outputs.append(
                SelectedCategorySignal(
                    source_module_id=self.module_id,
                    step_index=step_index,
                    payload={"category_id": self._selected_category, "created": True},
                    summary=f"Selected new category {self._selected_category}.",
                )
            )
        return outputs

    def get_state(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "adapter": self.adapter.model_key,
            "selected_category": self._selected_category,
            "category_count": len(getattr(self.model.art, "W", [])),
            "sample_count": self._sample_index,
            "pending_signals": len(self._inbox),
        }

    def get_output_signals(self) -> List[CompositionSignal]:
        outputs = self._output_signals
        self._output_signals = []
        return outputs

    def reset_runtime_state(self) -> None:
        self.trace_recorder.clear()
        self._inbox = []
        self._output_signals = []
        self._sample_index = 0
        self._selected_category = None
        self._last_trace_events = []
        self._build_runtime()

    def supports_expectation(self) -> bool:
        return False

    def explain_state(self) -> str:
        state = self.get_state()
        if state["selected_category"] is None:
            return f"{self.module_id} has not processed an input."
        return (
            f"{self.module_id} selected category {state['selected_category']} "
            f"from {state['category_count']} learned categories."
        )
