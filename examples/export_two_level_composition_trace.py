"""Export the two-level Fuzzy ART graph trace as JSON."""
from __future__ import annotations

from pathlib import Path

from two_level_fuzzy_art_pipeline import run_pipeline


def main() -> None:
    graph = run_pipeline()
    output_path = Path(__file__).parent / "output" / "two_level_composition_trace.json"
    graph.export_event_log_json(output_path)
    print(f"\nExported {len(graph.get_event_log())} events to {output_path}")


if __name__ == "__main__":
    main()
