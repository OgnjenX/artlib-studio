"""Feed-forward transforms for composition signals."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .signals import (
    CategoryActivationSignal,
    CompositionSignal,
    InputSignal,
    ExpectationSignal,
    SelectedCategorySignal,
)
from .association_memory import AssociationMemory


def selected_category_to_one_hot(
    signal: CompositionSignal,
    vector_size: int = 8,
) -> Optional[InputSignal]:
    """Convert a selected category to a fixed-width one-hot input."""
    if not isinstance(signal, SelectedCategorySignal):
        return None
    category_id = int(signal.payload["category_id"])
    if category_id < 0 or category_id >= vector_size:
        raise ValueError(
            f"category_id {category_id} does not fit one-hot size {vector_size}"
        )
    vector = [0.0] * vector_size
    vector[category_id] = 1.0
    return InputSignal(
        source_module_id=signal.source_module_id,
        step_index=signal.step_index,
        payload={
            "input": vector,
            "category_id": category_id,
            "representation": "one_hot",
        },
        summary=f"Category {category_id} encoded as one-hot input {vector}.",
    )


def selected_category_to_scalar_vector(
    signal: CompositionSignal,
    max_category_id: int = 7,
) -> Optional[InputSignal]:
    """Convert a selected category to a normalized one-value vector."""
    if not isinstance(signal, SelectedCategorySignal):
        return None
    if max_category_id < 1:
        raise ValueError("max_category_id must be at least 1")
    category_id = int(signal.payload["category_id"])
    if category_id < 0 or category_id > max_category_id:
        raise ValueError(
            f"category_id {category_id} exceeds maximum {max_category_id}"
        )
    vector = [category_id / max_category_id]
    return InputSignal(
        source_module_id=signal.source_module_id,
        step_index=signal.step_index,
        payload={
            "input": vector,
            "category_id": category_id,
            "representation": "scalar_vector",
        },
        summary=f"Category {category_id} encoded as scalar input {vector}.",
    )


def selected_category_to_activation_vector(
    signal: CompositionSignal,
) -> Optional[InputSignal]:
    """Forward a complete activation vector when an adapter supplies one."""
    if not isinstance(signal, CategoryActivationSignal):
        return None
    values = signal.payload.get("activation_vector")
    if values is None:
        values = signal.payload.get("choice_scores")
    if values is None:
        return None
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("activation vector must contain finite values")
    minimum = float(vector.min())
    maximum = float(vector.max())
    if minimum < 0.0 or maximum > 1.0:
        span = maximum - minimum
        vector = np.zeros_like(vector) if span == 0.0 else (vector - minimum) / span
    encoded = vector.tolist()
    return InputSignal(
        source_module_id=signal.source_module_id,
        step_index=signal.step_index,
        payload={"input": encoded, "representation": "activation_vector"},
        summary=f"Category activations forwarded as input {encoded}.",
    )


def high_category_to_expectation(
    signal: CompositionSignal,
    association_memory: AssociationMemory,
) -> Optional[ExpectationSignal]:
    """Convert a high-level selection into an association-backed expectation."""
    if not isinstance(signal, SelectedCategorySignal):
        return None
    high_category = int(signal.payload["category_id"])
    expected = sorted(
        association_memory.get_expected_low_level_categories(high_category)
    )
    if expected:
        explanation = (
            f"High-level category {high_category} expects low-level categories "
            f"{expected}."
        )
    else:
        explanation = (
            f"High-level category {high_category} has no learned low-level "
            "expectation yet."
        )
    return ExpectationSignal(
        source_module_id=signal.source_module_id,
        step_index=signal.step_index,
        expected_category_id=expected[0] if len(expected) == 1 else None,
        confidence=1.0 if expected else 0.0,
        payload={
            "high_level_category_id": high_category,
            "expected_category_ids": expected,
        },
        summary=explanation,
        explanation=explanation,
    )
