"""Model adapter interface for pluggable ARTLib models."""
from typing import Any, Dict, List, Optional, Set
from abc import ABC, abstractmethod
from .capabilities import Capability
from .recorder import TraceRecorder


class ARTModelAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the adapter (e.g. 'Fuzzy ART')."""
        pass

    @property
    @abstractmethod
    def model_key(self) -> str:
        """Identifier key for the adapter (e.g. 'fuzzy_art')."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> Set[Capability]:
        """Set of capabilities supported by this adapter."""
        pass

    def supports(self, capability: Capability) -> bool:
        """Check if a specific capability is supported."""
        return capability in self.capabilities

    @abstractmethod
    def create_model(self, params: dict) -> Any:
        """Create and return the underlying ART model."""
        pass

    @abstractmethod
    def default_params(self) -> dict:
        """Return default parameters for the model."""
        pass

    @abstractmethod
    def param_schema(self) -> dict:
        """Return parameter schema for UI generation."""
        pass

    def prepare_data(self, X: Any) -> Any:
        """Prepare data appropriately for the model."""
        return X

    @abstractmethod
    def fit_with_trace(self, X: Any, params: dict, recorder: Optional[TraceRecorder] = None) -> Any:
        """Fit the model and record trace to the given recorder."""
        pass

    @abstractmethod
    def predict(self, model: Any, X: Any) -> Any:
        """Run predictions on the model."""
        pass

    def get_category_geometry_2d(self, model: Any) -> Any:
        """Return 2D category geometry if supported."""
        raise NotImplementedError("CATEGORY_BOXES_2D not supported or implemented.")

    def get_category_prototypes(self, model: Any) -> Any:
        """Return prototypes/weights from the model."""
        raise NotImplementedError("CATEGORY_PROTOTYPES not supported or implemented.")

    def build_competition_table(self, trace_events: List[Any], sample_index: int) -> Any:
        """Build data for choice/match visualization."""
        raise NotImplementedError("Competition table visualization not supported.")

    def explain_event(self, event: Any) -> str:
        """Return a human-readable explanation of an event."""
        return f"Event {event.type.name}"

