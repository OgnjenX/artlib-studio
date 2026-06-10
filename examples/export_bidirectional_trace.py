"""Export the bidirectional expectation trace as JSON."""
from __future__ import annotations

from pathlib import Path

from bidirectional_expectation_demo import run_demo


def main() -> None:
    graph = run_demo()
    output_path = (
        Path(__file__).parent
        / "output"
        / "bidirectional_composition_trace.json"
    )
    graph.export_event_log_json(output_path)
    print(f"\nExported {len(graph.get_event_log())} events to {output_path}")


if __name__ == "__main__":
    main()
