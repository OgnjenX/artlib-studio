"""Export the strict-context modulation trace."""
from __future__ import annotations

from pathlib import Path

from modulatory_vigilance_demo import run_modulatory_demo
from artlib_studio.composition.events import CompositionEventType


def main() -> None:
    _, graph = run_modulatory_demo()
    output_path = Path(__file__).parent / "output" / "modulatory_trace.json"
    graph.export_event_log_json(output_path)
    event_types = {event.type for event in graph.get_event_log()}
    required = {
        CompositionEventType.MODULATION_SENT,
        CompositionEventType.MODULATION_RECEIVED,
        CompositionEventType.MODULE_PARAMETER_MODULATED,
        CompositionEventType.MODULE_PARAMETER_RESTORED,
        CompositionEventType.MODULE_SELECTED_CATEGORY,
    }
    missing = required - event_types
    if missing:
        names = ", ".join(sorted(event.value for event in missing))
        raise RuntimeError(f"Exported trace is missing required events: {names}")
    print(f"Exported {len(graph.get_event_log())} events to {output_path}")


if __name__ == "__main__":
    main()
