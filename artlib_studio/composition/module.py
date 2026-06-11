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
    ModulatorySignal,
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
        self._step_param_snapshot: Dict[str, Dict[str, Any]] = {}
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
            if isinstance(signal, ModulatorySignal):
                self.apply_modulation(signal)
        for signal in pending:
            if isinstance(signal, InputSignal):
                self._process_input(signal)

    def get_params(self) -> Dict[str, Any]:
        return dict(self.model.art.params)

    def set_param(self, name: str, value: Any, persistent: bool = False) -> None:
        if name not in self.model.art.params:
            raise ValueError(
                f"Module {self.module_id!r} does not support parameter {name!r}"
            )
        if name == "rho" and not 0.0 <= float(value) <= 1.0:
            raise ValueError("rho modulation must remain in [0, 1]")
        self.model.art.params[name] = value
        if persistent:
            self.model_params[name] = value

    def apply_modulation(self, signal: ModulatorySignal) -> Dict[str, Any]:
        name = signal.target_param
        params = self.get_params()
        if name not in params:
            raise ValueError(
                f"Module {self.module_id!r} does not support parameter {name!r}"
            )
        before = params[name]
        if signal.mode == "set":
            after = signal.value
        elif signal.mode == "add":
            after = before + signal.value
        elif signal.mode == "multiply":
            after = before * signal.value
        else:
            raise ValueError(f"Unsupported modulation mode {signal.mode!r}")
        if signal.duration == "current_step":
            self._step_param_snapshot.setdefault(
                name,
                {
                    "original_value": before,
                    "source_module_id": signal.source_module_id,
                    "target_param": name,
                    "mode": signal.mode,
                    "duration": signal.duration,
                    "explanation": signal.explanation,
                },
            )
        elif signal.duration != "persistent":
            raise ValueError(
                f"Unsupported modulation duration {signal.duration!r}"
            )
        self.set_param(
            name,
            after,
            persistent=signal.duration == "persistent",
        )
        return {
            "target_param": name,
            "mode": signal.mode,
            "value": signal.value,
            "duration": signal.duration,
            "before": before,
            "after": after,
            "explanation": signal.explanation,
        }

    def restore_step_modulations(self) -> List[Dict[str, Any]]:
        restored = []
        for name, snapshot in self._step_param_snapshot.items():
            before = self.model.art.params[name]
            restored_value = snapshot["original_value"]
            self.set_param(name, restored_value)
            restored.append(
                {
                    **snapshot,
                    "before": before,
                    "after": restored_value,
                    "restored_value": restored_value,
                }
            )
        self._step_param_snapshot = {}
        return restored

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
        prepared_batch = np.atleast_2d(prepared)
        if hasattr(self.model.art, "validate_data"):
            self.model.art.validate_data(prepared_batch)
        if hasattr(self.model.art, "check_dimensions"):
            self.model.art.check_dimensions(prepared_batch)
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
        selected_payload: Optional[Dict[str, Any]] = None
        selected_summary = ""
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
                selected_payload = payload
                selected_summary = summary
            elif event.type == EventType.MATCH_TEST:
                outputs.append(MatchSignal(**common))
            elif event.type == EventType.RESET:
                outputs.append(ResetSignal(**common))
            elif event.type == EventType.RESONANCE:
                outputs.append(ResonanceSignal(**common))
            elif event.type == EventType.LEARNING:
                outputs.append(LearningSignal(**common))
            elif event.type == EventType.CATEGORY_CREATED:
                selected_payload = {
                    **payload,
                    "category_id": payload["created_index"],
                    "created": True,
                }
                selected_summary = (
                    f"Created and selected category {payload['created_index']}."
                )

        outputs.append(
            SelectedCategorySignal(
                source_module_id=self.module_id,
                step_index=step_index,
                payload=selected_payload
                or {"category_id": self._selected_category},
                summary=selected_summary
                or f"Selected category {self._selected_category}.",
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
            "params": self.get_params(),
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
        self._step_param_snapshot = {}
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
