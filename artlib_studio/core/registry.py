"""Adapter registry for ARTLib Studio."""
from typing import Dict, List
from .model_adapter import ARTModelAdapter

_ADAPTERS: Dict[str, ARTModelAdapter] = {}

def register_adapter(adapter: ARTModelAdapter):
    """Register an ART model adapter."""
    _ADAPTERS[adapter.model_key] = adapter

def get_adapter(model_key: str) -> ARTModelAdapter:
    """Get an adapter by its model key."""
    if model_key not in _ADAPTERS:
        raise KeyError(f"Adapter with model_key '{model_key}' not found.")
    return _ADAPTERS[model_key]

def list_adapters() -> List[ARTModelAdapter]:
    """List all registered adapters."""
    return list(_ADAPTERS.values())

# Auto-register default adapters.
from ..adapters.fuzzy_art import FuzzyARTAdapter
from ..adapters.gaussian_art import GaussianARTAdapter
from ..adapters.hypersphere_art import HypersphereARTAdapter

register_adapter(FuzzyARTAdapter())
register_adapter(GaussianARTAdapter())
register_adapter(HypersphereARTAdapter())
