"""Capabilities system for ARTLib Studio.

Capabilities express what visualizations are safe and meaningful for a given model.
"""
from enum import Enum, auto

class Capability(Enum):
    TRACE_EXECUTION = auto()
    CHOICE_SCORES = auto()
    MATCH_SCORES = auto()
    VIGILANCE = auto()
    RESET_SEARCH = auto()
    LEARNING_UPDATES = auto()
    WEIGHT_BEFORE_AFTER = auto()
    COMPLEMENT_CODING = auto()
    CATEGORY_BOXES_2D = auto()
    CATEGORY_PROTOTYPES = auto()
    GAUSSIAN_REGIONS_2D = auto()
    HYPERSPHERE_REGIONS_2D = auto()
    SUPERVISED_MAPPING = auto()
    MULTI_CHANNEL_INPUT = auto()
    STREAMLIT_EXPLORER = auto()
