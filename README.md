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
streamlit run artlib_studio/apps/streamlit_app.py
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

## How to use

See the [Streamlit app](https://github.com/NiklasMelton/ARTLibStudio/blob/master/artlib_studio/apps/streamlit_app.py) for an interactive demo.
