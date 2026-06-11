"""Bounded discrete scheduler for ART composition graphs."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple, TYPE_CHECKING

from .edges import EdgeType
from .events import CompositionEvent, CompositionEventType
from .signals import (
    CompositionSignal,
    ExpectationSignal,
    InputSignal,
    LearningSignal,
    ModulatorySignal,
    ResetSignal,
    ResonanceSignal,
    SelectedCategorySignal,
)

if TYPE_CHECKING:
    from .graph import ARTCompositionGraph


class DiscreteScheduler:
    def __init__(self, max_settling_steps: int = 5):
        if max_settling_steps < 1:
            raise ValueError("max_settling_steps must be at least 1")
        self.max_settling_steps = max_settling_steps
        self.step_index = 0

    def run(
        self,
        graph: "ARTCompositionGraph",
        input_signals: Iterable[CompositionSignal],
    ) -> bool:
        external_signals = list(input_signals)
        # Context modulation is routed first so recipients use the changed
        # parameters when processing external input in this graph step.
        pending = self._initial_context_signals(graph) + external_signals
        pending_associations: Set[Tuple[int, int]] = set()
        for settling_step in range(self.max_settling_steps):
            if not pending:
                for high_category, low_category in pending_associations:
                    graph.association_memory.record_pair(
                        high_category,
                        low_category,
                    )
                self._restore_step_modulations(graph)
                graph._record(
                    CompositionEvent(
                        CompositionEventType.GRAPH_SETTLED,
                        self.step_index,
                        payload={"settling_steps": settling_step},
                    )
                )
                self.step_index += 1
                return True

            recipients: Dict[str, List[CompositionSignal]] = defaultdict(list)
            for signal in pending:
                targets = (
                    [signal.target_module_id]
                    if signal.target_module_id is not None
                    else list(graph.modules)
                )
                for target in targets:
                    module = graph.get_module(target)
                    delivered = signal.for_target(target)
                    graph._record(
                        CompositionEvent(
                            CompositionEventType.SIGNAL_RECEIVED,
                            self.step_index,
                            module_id=target,
                            payload={
                                "source_module_id": delivered.source_module_id,
                                "target_module_id": target,
                                "signal_type": type(delivered).__name__,
                                "signal_payload": dict(delivered.payload),
                                "summary": delivered.summary,
                            },
                        )
                    )
                    if isinstance(delivered, ModulatorySignal):
                        self._apply_modulation(graph, target, delivered)
                        continue

                    module.receive(delivered)
                    recipients[target].append(delivered)
                    if isinstance(delivered, InputSignal):
                        graph._record(
                            CompositionEvent(
                                CompositionEventType.MODULE_RECEIVED_INPUT,
                                self.step_index,
                                module_id=target,
                                payload={
                                    **dict(delivered.payload),
                                    "source_module_id": delivered.source_module_id,
                                },
                            )
                        )
                    elif isinstance(delivered, ExpectationSignal):
                        self._evaluate_expectation(graph, target, delivered)

            emitted: List[CompositionSignal] = []
            for module_id in recipients:
                module = graph.get_module(module_id)
                module.step()
                outputs = module.get_output_signals()
                emitted.extend(outputs)
                self._record_module_outputs(graph, module_id, outputs)
                pending_associations.update(
                    self._association_pairs_for_outputs(graph, module_id, outputs)
                )

            pending = []
            for signal in emitted:
                for edge in graph.outgoing_edges(signal.source_module_id):
                    transmitted = edge.transmit(signal)
                    if transmitted is None:
                        continue
                    pending.append(transmitted)
                    graph._record(
                        CompositionEvent(
                            CompositionEventType.SIGNAL_SENT,
                            self.step_index,
                            module_id=signal.source_module_id,
                            payload={
                                "target_module_id": edge.target_module_id,
                                "edge_type": edge.edge_type.value,
                                "signal_type": type(transmitted).__name__,
                                "source_signal_type": type(signal).__name__,
                                "signal_payload": dict(transmitted.payload),
                                "transform_name": edge.transform_name,
                                "summary": transmitted.summary,
                            },
                        )
                    )
                    if isinstance(transmitted, ExpectationSignal):
                        graph._record(
                            CompositionEvent(
                                CompositionEventType.EXPECTATION_SENT,
                                self.step_index,
                                module_id=signal.source_module_id,
                                payload={
                                    "target_module_id": edge.target_module_id,
                                    **dict(transmitted.payload),
                                    "confidence": transmitted.confidence,
                                    "explanation": transmitted.explanation,
                                },
                            )
                        )
                    elif isinstance(transmitted, ModulatorySignal):
                        graph._record(
                            CompositionEvent(
                                CompositionEventType.MODULATION_SENT,
                                self.step_index,
                                module_id=signal.source_module_id,
                                payload={
                                    "target_module_id": edge.target_module_id,
                                    **dict(transmitted.payload),
                                },
                            )
                        )

        self._restore_step_modulations(graph)
        graph._record(
            CompositionEvent(
                CompositionEventType.GRAPH_FAILED_TO_SETTLE,
                self.step_index,
                payload={"max_settling_steps": self.max_settling_steps},
            )
        )
        self.step_index += 1
        return False

    def _initial_context_signals(
        self,
        graph: "ARTCompositionGraph",
    ) -> List[CompositionSignal]:
        transmitted_signals: List[CompositionSignal] = []
        for module_id, module in graph.modules.items():
            emit = getattr(module, "emit_modulations", None)
            if emit is None:
                continue
            for signal in emit(self.step_index):
                for edge in graph.outgoing_edges(module_id):
                    if edge.edge_type != EdgeType.MODULATORY:
                        continue
                    transmitted = edge.transmit(signal)
                    if transmitted is None:
                        continue
                    transmitted_signals.append(transmitted)
                    graph._record(
                        CompositionEvent(
                            CompositionEventType.SIGNAL_SENT,
                            self.step_index,
                            module_id=module_id,
                            payload={
                                "target_module_id": edge.target_module_id,
                                "edge_type": edge.edge_type.value,
                                "signal_type": type(transmitted).__name__,
                                "source_signal_type": type(signal).__name__,
                                "signal_payload": dict(transmitted.payload),
                                "transform_name": edge.transform_name,
                                "summary": transmitted.summary,
                            },
                        )
                    )
                    graph._record(
                        CompositionEvent(
                            CompositionEventType.MODULATION_SENT,
                            self.step_index,
                            module_id=module_id,
                            payload={
                                "target_module_id": edge.target_module_id,
                                **dict(transmitted.payload),
                            },
                        )
                    )
        return transmitted_signals

    def _apply_modulation(
        self,
        graph: "ARTCompositionGraph",
        target_module_id: str,
        signal: ModulatorySignal,
    ) -> None:
        module = graph.get_module(target_module_id)
        apply = getattr(module, "apply_modulation", None)
        if apply is None:
            raise ValueError(
                f"Module {target_module_id!r} does not support modulation"
            )
        details = apply(signal)
        payload = {
            "source_module_id": signal.source_module_id,
            "target_module_id": target_module_id,
            **details,
        }
        graph._record(
            CompositionEvent(
                CompositionEventType.MODULATION_RECEIVED,
                self.step_index,
                module_id=target_module_id,
                payload=payload,
            )
        )
        graph._record(
            CompositionEvent(
                CompositionEventType.MODULE_PARAMETER_MODULATED,
                self.step_index,
                module_id=target_module_id,
                payload=payload,
            )
        )

    def _restore_step_modulations(self, graph: "ARTCompositionGraph") -> None:
        for module_id, module in graph.modules.items():
            restore = getattr(module, "restore_step_modulations", None)
            if restore is None:
                continue
            for details in restore():
                graph._record(
                    CompositionEvent(
                        CompositionEventType.MODULE_PARAMETER_RESTORED,
                        self.step_index,
                        module_id=module_id,
                        payload={
                            "target_module_id": module_id,
                            **details,
                        },
                    )
                )

    def _association_pairs_for_outputs(
        self,
        graph: "ARTCompositionGraph",
        module_id: str,
        outputs: List[CompositionSignal],
    ) -> Set[Tuple[int, int]]:
        high_categories = [
            int(signal.payload["category_id"])
            for signal in outputs
            if isinstance(signal, SelectedCategorySignal)
        ]
        if not high_categories:
            return set()

        pairs = set()
        top_down_targets = {
            edge.target_module_id
            for edge in graph.outgoing_edges(module_id)
            if edge.edge_type == EdgeType.TOP_DOWN_EXPECTATION
        }
        for low_module_id in top_down_targets:
            low_category = graph.get_module(low_module_id).get_state().get(
                "selected_category"
            )
            if low_category is not None:
                pairs.add((high_categories[-1], int(low_category)))
        return pairs

    def _evaluate_expectation(
        self,
        graph: "ARTCompositionGraph",
        target_module_id: str,
        signal: ExpectationSignal,
    ) -> None:
        expected = signal.expected_category_ids
        current = graph.get_module(target_module_id).get_state().get(
            "selected_category"
        )
        explanation = signal.explanation or signal.summary or ""
        if not expected:
            matched = None
            explanation = (
                explanation
                or "No prior expectation is available for this high-level category."
            )
            payload = {
                **dict(signal.payload),
                "source_module_id": signal.source_module_id,
                "target_module_id": target_module_id,
                "current_low_level_category_id": current,
                "matched": matched,
                "confidence": signal.confidence,
                "explanation": explanation,
            }
            graph._record(
                CompositionEvent(
                    CompositionEventType.EXPECTATION_RECEIVED,
                    self.step_index,
                    module_id=target_module_id,
                    payload=payload,
                )
            )
            graph._record(
                CompositionEvent(
                    CompositionEventType.EXPECTATION_UNAVAILABLE,
                    self.step_index,
                    module_id=target_module_id,
                    payload=payload,
                )
            )
            graph._record(
                CompositionEvent(
                    CompositionEventType.CROSS_MODULE_EXPECTATION_UNKNOWN,
                    self.step_index,
                    payload=payload,
                )
            )
            return

        matched = current is not None and int(current) in expected
        explanation = (
            explanation
            or (
                "Expectation matched."
                if matched
                else "Expectation was violated."
            )
        )
        payload = {
            **dict(signal.payload),
            "source_module_id": signal.source_module_id,
            "target_module_id": target_module_id,
            "current_low_level_category_id": current,
            "matched": matched,
            "confidence": signal.confidence,
            "explanation": explanation,
        }
        graph._record(
            CompositionEvent(
                CompositionEventType.EXPECTATION_RECEIVED,
                self.step_index,
                module_id=target_module_id,
                payload=payload,
            )
        )
        graph._record(
            CompositionEvent(
                (
                    CompositionEventType.EXPECTATION_MATCHED
                    if matched
                    else CompositionEventType.EXPECTATION_MISMATCHED
                ),
                self.step_index,
                module_id=target_module_id,
                payload=payload,
            )
        )
        graph._record(
            CompositionEvent(
                (
                    CompositionEventType.CROSS_MODULE_RESONANCE
                    if matched
                    else CompositionEventType.CROSS_MODULE_MISMATCH
                ),
                self.step_index,
                payload=payload,
            )
        )

    def _record_module_outputs(
        self,
        graph: "ARTCompositionGraph",
        module_id: str,
        outputs: List[CompositionSignal],
    ) -> None:
        event_types = {
            SelectedCategorySignal: CompositionEventType.MODULE_SELECTED_CATEGORY,
            ResonanceSignal: CompositionEventType.MODULE_RESONATED,
            ResetSignal: CompositionEventType.MODULE_RESET,
            LearningSignal: CompositionEventType.MODULE_LEARNED,
        }
        for signal in outputs:
            event_type = event_types.get(type(signal))
            if event_type is not None:
                graph._record(
                    CompositionEvent(
                        event_type,
                        self.step_index,
                        module_id=module_id,
                        payload=dict(signal.payload),
                    )
                )
            if (
                isinstance(signal, SelectedCategorySignal)
                and signal.payload.get("created")
            ):
                graph._record(
                    CompositionEvent(
                        CompositionEventType.MODULE_CATEGORY_CREATED,
                        self.step_index,
                        module_id=module_id,
                        payload=dict(signal.payload),
                    )
                )
