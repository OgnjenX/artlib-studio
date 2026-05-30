# Execution Explorer — design notes

This document explains how the v0.2 Execution Explorer exposes Fuzzy ART
internal execution to a human user.

## What the Execution Explorer Shows

The Execution Explorer makes Fuzzy ART understandable by breaking down the execution steps. It focuses on the internal algorithm dynamics such as category activation, match threshold tests, reset routines, and category formation. A user sees exact choice scores, match tracking values, and detailed reasons about why a category won or failed.

## Event types

Event types produced by the instrumentation layer (`artlib_studio.events`):

- **INPUT_RECEIVED**: an input sample has been presented to the ART module.
- **CATEGORY_EVALUATED**: a category's choice/activation value was computed.
- **CATEGORY_SELECTED**: a candidate was taken from the sorted list for match testing.
- **MATCH_TEST**: a match value and the vigilance threshold are compared; payload
  includes match score and whether it passed.
- **RESET**: a candidate failed a match test and a reset (or match-tracking) step
  was performed.
- **RESONANCE**: a candidate passed the match test and was accepted.
- **LEARNING**: the winner updated its weights (learning step completed).
- **CATEGORY_CREATED**: a new category/weight was created.

Each event includes a timestamp, a `sample_index`, a `explanation` string, and a payload with relevant numeric/contextual
information so the UI can quickly present human-readable logs of what occurred.

## Interpreting ART Execution
- **Vigilance**: The minimum match threshold required for a category to incorporate an input pattern. If vigilance is high, categories remain small, and many are automatically generated.
- **Search & Reset**: If the highest activated/choice category fails the vigilance match test, a "Reset" event fires, and the next highest scored category is selected until a "Resonance" match is found. 
- **Resonance & Category Creation**: A successful match leads to "Resonance" and "Learning", adapting the category prototype box to encompass the new input. If no pre-existing categories match, a "Category Created" happens.

## Running Examples

### Step-by-Step CLI Trace
View the human-readable event execution formatted to terminal:
```bash
python examples/trace_fuzzy_art_step_by_step.py
```

### Trace to JSON Export
You can export the traces to JSON format in the `examples/output/` directory:
```bash
python examples/export_fuzzy_art_trace.py
```

### Streamlit UI Prototype
Launch the local web interactive debugger:
```bash
streamlit run artlib_studio/streamlit_app.py
```

UI Features:
- Left panel provides parameter controls to generate datasets and re-execute FuzzyART traces.
- Main panel shows a dynamic scatter plot matching the current sample overlay with bounding box visual proxies.
- Right panel supplies chronological playback actions (`<< Step Prev Event`, `Step Next Event >>`) alongside textual payload explanations.

## Implementation Notes

- The `InstrumentedART` wrapper deliberately replays `BaseART.step_fit` logic and
  calls underlying methods iteratively without interfering with math.
- We added an `explanation` natural-language string to every trace dict payload to immediately assist UI and logging visibility efforts.
