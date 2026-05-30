# Execution Explorer — design notes

This document explains how the v0.3 Execution Explorer exposes Fuzzy ART
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

## ART Internals Inspector (v0.3)

The ARTLib Studio v0.3 introduces a dedicated Internals Inspector to help users understand the "why" behind each decision.

### Key Components:

- **Choice Score Visualization**: A bar chart showing the competition between all categories. The winning category (highest choice score) is highlighted.
- **Match vs Vigilance**: A clear comparison of the current category's match score against the global vigilance threshold.
- **Search History**: A step-by-step summary of the search process for the current sample, showing which categories were tested and reset before resonance or creation.
- **Category Competition Table**: Detailed data for every category, including choice score, match score, and final result (resonance, reset, or not selected).
- **Learning Update View**: When resonance occurs, the inspector shows the weights before and after learning, highlighting the change (delta).
- **Conceptual ART Circuit**: A simplified diagram showing the flow of information from input (F1) to category selection (F2) and match comparison.

### Understanding the Visualization

- **Choice Score**: Measures how much a category "wants" the current input. It's often influenced by the size of the category (smaller categories are preferred when scores are otherwise tied).
- **Match Score**: Measures how well the input "fits" inside the category's learned box.
- **Vigilance (ρ)**: The strictness of the match requirement.
- **Reset**: Occurs when a category's match score is below vigilance. The category is temporarily disabled, and the search continues.
- **Resonance**: Occurs when a category passes the vigilance test. This triggers weight adaptation (learning).
- **Approximate Category Expectations**: The bounding boxes in the main plot represent the learned weights. Note that these are approximations; the actual decision is made by the competitive choice and match scores.

## Implementation Notes

- The `InstrumentedART` wrapper deliberately replays `BaseART.step_fit` logic and
  calls underlying methods iteratively without interfering with math.
- We added an `explanation` natural-language string to every trace dict payload to immediately assist UI and logging visibility efforts.
