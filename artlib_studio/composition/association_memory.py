"""Association memory for simplified top-down category expectations."""
from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List, Set


class AssociationMemory:
    """Track observed high-level to low-level category associations."""

    def __init__(self) -> None:
        self._associations: DefaultDict[int, Set[int]] = defaultdict(set)

    def record_pair(self, high_level_category: int, low_level_category: int) -> None:
        self._associations[int(high_level_category)].add(int(low_level_category))

    def get_expected_low_level_categories(
        self,
        high_level_category: int,
    ) -> Set[int]:
        return set(self._associations.get(int(high_level_category), set()))

    def match_expectation(
        self,
        high_level_category: int,
        current_low_level_category: int,
    ) -> bool:
        return int(current_low_level_category) in self.get_expected_low_level_categories(
            high_level_category
        )

    def to_dict(self) -> Dict[int, List[int]]:
        return {
            category: sorted(expected)
            for category, expected in sorted(self._associations.items())
        }
