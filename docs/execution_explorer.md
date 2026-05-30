# Execution Explorer — design notes

This document explains how the v0.1 Execution Explorer exposes Fuzzy ART
internal execution to a human user.

Event types produced by the instrumentation layer (artlib_studio.events):

- INPUT_RECEIVED: an input sample has been presented to the ART module.
- CATEGORY_EVALUATED: a category's choice/activation value was computed.
- CATEGORY_SELECTED: a candidate was taken from the sorted list for match testing.
- MATCH_TEST: a match value and the vigilance threshold are compared; payload
  includes match score and whether it passed.
- RESET: a candidate failed a match test and a reset (or match-tracking) step
  was performed.
- RESONANCE: a candidate passed the match test and was accepted.
- LEARNING: the winner updated its weights (learning step completed).
- CATEGORY_CREATED: a new category/weight was created.

Each event includes a timestamp and a payload with relevant numeric/contextual
information so the UI can present an explanation.

UI Responsibilities:

- Panel A: display the raw input currently being inspected (in original input
  space, not complement coded).
- Panel B: render all categories (bounding boxes for Fuzzy ART when 2D).
- Panel C: highlight the currently selected/winning category.
- Panel D: show current match score (numerical) and an explanation.
- Panel E: show current vigilance value (rho) and allow a slider to change it
  (for immediate feedback we can re-run the fit with a new rho).
- Panel F: timeline of events allowing stepping forward/backward and selecting
  any event to inspect its payload.

Implementation notes:

- The InstrumentedART wrapper deliberately replays BaseART.step_fit logic and
  calls underlying category_choice/match_criterion/update/new_weight so that
  the wrapped ART object performs the actual math. The wrapper only observes
  and records events.
- The Streamlit app is a minimal first UI; future work will add richer
  explanations and an educational mode that generates natural-language
  descriptions for events.

