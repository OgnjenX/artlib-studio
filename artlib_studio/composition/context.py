"""Explicit context nodes that emit parameter modulation signals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .module import ComposableARTModule
from .signals import CompositionSignal, ModulatorySignal


@dataclass(frozen=True)
class ContextRule:
    name: str
    active: bool = True
    target_param: str = "rho"
    mode: str = "set"
    value: float = 0.85
    duration: str = "current_step"
    explanation: str = ""


class ContextModule(ComposableARTModule):
    """Non-learning module that emits active context rules each graph step."""

    is_context_module = True

    def __init__(self, module_id: str, rules: List[ContextRule]):
        super().__init__(module_id)
        self.rules = list(rules)
        self._outputs: List[CompositionSignal] = []
        self._step_index = 0

    def receive(self, signal: CompositionSignal) -> None:
        self._step_index = signal.step_index

    def step(self) -> None:
        self._outputs = self.emit_modulations(self._step_index)

    def emit_modulations(self, step_index: int) -> List[ModulatorySignal]:
        return [
            ModulatorySignal(
                source_module_id=self.module_id,
                step_index=step_index,
                target_param=rule.target_param,
                mode=rule.mode,
                value=float(rule.value),
                duration=rule.duration,
                payload={
                    "rule_name": rule.name,
                    "target_param": rule.target_param,
                    "mode": rule.mode,
                    "value": float(rule.value),
                    "duration": rule.duration,
                    "explanation": rule.explanation,
                },
                summary=rule.explanation or f"Context rule {rule.name}.",
                explanation=rule.explanation or f"Context rule {rule.name}.",
            )
            for rule in self.rules
            if rule.active
        ]

    def get_state(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "type": "context",
            "active_rules": [rule.name for rule in self.rules if rule.active],
            "rule_count": len(self.rules),
            "pending_signals": 0,
        }

    def get_output_signals(self) -> List[CompositionSignal]:
        outputs = self._outputs
        self._outputs = []
        return outputs

    def reset_runtime_state(self) -> None:
        self._outputs = []
        self._step_index = 0

    def supports_expectation(self) -> bool:
        return False

    def explain_state(self) -> str:
        return (
            f"{self.module_id} has "
            f"{len([rule for rule in self.rules if rule.active])} active rules."
        )
