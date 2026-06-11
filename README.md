# ARTLib Studio

**ARTLib Studio** is an interactive execution explorer and instrumentation framework for [AdaptiveResonanceLib](https://github.com/NiklasMelton/AdaptiveResonanceLib).

## Current status:

* FuzzyART adapter
* Gaussian ART adapter
* Hypersphere ART adapter
* adapter registry
* capability-aware visualization
* trace recorder
* Streamlit explorer
* ART Composition Studio
* YAML/JSON graph configuration
* context-driven vigilance modulation

## Planned:

* ART1
* TopoART
* ARTMAP / supervised models

## Prerequisites
* Python >= 3.11
* [NumPy](https://numpy.org/)
* [matplotlib](https://matplotlib.org/)
* [scikit-learn](https://scikit-learn.org/)
* [streamlit](https://streamlit.io/)

## Setup & Installation

Clone both `AdaptiveResonanceLib` and `artlib-studio` into the same parent directory (or adjust paths accordingly). From inside `artlib-studio`:

```bash
pip install -e ../AdaptiveResonanceLib
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Running the Explorer

```bash
uv run streamlit run artlib_studio/apps/streamlit_app.py
```

## ARTLib Studio Current Status

Currently implemented features:

* Adapter/capability architecture
* FuzzyARTAdapter as the first supported adapter
* InstrumentedART wrapper for BaseART models
* Execution trace recording to memory and JSON
* Educational step-by-step Streamlit app
* Modular event-based visualization architecture
* Live event tracker
* Timeline replay
* Final-state view toggle
* Processed/future sample distinction
* Category creation timeline tracking
* Event-aware plot highlighting
* Match/vigilance panel
* Category competition table
* Search history panel
* Current-step explanation cards
* ART composition protocol
* Two-level feed-forward Fuzzy ART pipeline demo
* Graph-level composition event trace and JSON export
* Bidirectional association-based expectation prototype
* Cross-module resonance and mismatch tracing
* Streamlit composition graph inspection
* YAML/JSON composition graph loader and builder
* Config-driven graph composer with static preview
* Context nodes and `MODULATORY` vigilance edges
* Current-step and persistent `rho` modulation
* Same-step modulation tracing and parameter restoration

## Composition Studio

The Composition Studio provides three built-in educational experiments:

* Two-level feed-forward Fuzzy ART
* Bidirectional association-backed expectation
* Modulatory vigilance context

It also includes a form-based graph composer for adding modules, typed edges,
transforms, and association entries. Graph configurations can be imported or
exported as YAML or JSON, and graph traces can be exported as JSON.

Run the config and modulation examples with:

```bash
uv run python examples/run_composition_config.py examples/configs/two_level_fuzzy_art.yaml
uv run python examples/run_composition_config.py examples/configs/bidirectional_expectation.yaml
uv run python examples/modulatory_vigilance_demo.py
uv run python examples/export_modulatory_trace.py
```

These experiments are software prototypes for studying ART composition. They
are not full Grossberg biological neural circuits, synchronized neural
dynamics, or learned top-down ART templates.

For `current_step` context rules, modulation is applied before ART input
processing and learning, then restored after the graph settles. Persistent
modulation remains active. The Composition Studio exposes this ordering in a
dedicated modulation event table. Context nodes are explicit educational
control nodes, not a biological neuromodulatory model.

## How to use

See the [Streamlit app](https://github.com/OgnjenX/artlib-studio/blob/main/artlib_studio/apps/streamlit_app.py) for the interactive explorer and Composition Studio.
