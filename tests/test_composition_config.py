import json
from pathlib import Path

import pytest

from artlib_studio.composition.builder import build_graph_from_config
from artlib_studio.composition.config import (
    graph_config_from_dict,
    load_graph_config,
    save_graph_config,
)
from artlib_studio.composition.events import CompositionEventType
from artlib_studio.composition.signals import InputSignal
from artlib_studio.apps.components.graph_composer import (
    add_edge_to_config,
    add_module_to_config,
    config_dict_to_json,
    config_dict_to_yaml,
    empty_graph_config_dict,
    remove_module_from_config,
    remove_edge_from_config,
)
from artlib_studio.visualization.composition_trace import (
    build_edge_table,
    build_graph_overview,
    build_module_state_table,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "examples" / "configs"


def test_load_yaml_and_json_configs():
    yaml_config = load_graph_config(CONFIG_DIR / "two_level_fuzzy_art.yaml")
    json_config = load_graph_config(CONFIG_DIR / "two_level_fuzzy_art.json")
    assert yaml_config.name == json_config.name
    assert len(yaml_config.modules) == 2


def test_config_validation_rejects_unknown_target_and_transform():
    base = {
        "name": "invalid",
        "modules": [
            {"id": "low", "type": "adapter", "adapter": "fuzzy_art"}
        ],
    }
    with pytest.raises(ValueError, match="unknown module"):
        graph_config_from_dict(
            {
                **base,
                "edges": [
                    {"source": "low", "target": "missing", "type": "BOTTOM_UP"}
                ],
            }
        )
    with pytest.raises(ValueError, match="Unsupported transform"):
        graph_config_from_dict(
            {
                **base,
                "edges": [
                    {
                        "source": "low",
                        "target": "low",
                        "type": "BOTTOM_UP",
                        "transform": {"name": "unknown"},
                    }
                ],
            }
        )


def test_config_round_trip(tmp_path):
    config = load_graph_config(CONFIG_DIR / "two_level_fuzzy_art.yaml")
    yaml_path = save_graph_config(config, tmp_path / "graph.yaml")
    json_path = save_graph_config(config, tmp_path / "graph.json")
    assert load_graph_config(yaml_path) == config
    assert load_graph_config(json_path) == config
    assert json.loads(json_path.read_text())["name"] == config.name


def test_two_level_config_builds_and_runs():
    config = load_graph_config(CONFIG_DIR / "two_level_fuzzy_art.yaml")
    graph = build_graph_from_config(config)
    graph.step(
        [
            InputSignal(
                "environment",
                target_module_id="low_level_fuzzy_art",
                payload={"input": [0.1, 0.2]},
            )
        ]
    )
    assert graph.get_module("high_level_fuzzy_art").get_state()["sample_count"] == 1


def test_bidirectional_config_emits_expectation_events():
    config = load_graph_config(CONFIG_DIR / "bidirectional_expectation.yaml")
    graph = build_graph_from_config(config)
    graph.step(
        [
            InputSignal(
                "environment",
                target_module_id="low_level_fuzzy_art",
                payload={"input": [0.1, 0.2]},
            )
        ]
    )
    types = {event.type for event in graph.get_event_log()}
    assert CompositionEventType.EXPECTATION_SENT in types
    assert CompositionEventType.CROSS_MODULE_EXPECTATION_UNKNOWN in types


def test_composer_helpers_create_reloadable_config():
    data = empty_graph_config_dict()
    data = add_module_to_config(
        data,
        {
            "id": "low",
            "type": "adapter",
            "adapter": "fuzzy_art",
            "params": {"rho": 0.8},
        },
    )
    data = add_module_to_config(
        data,
        {
            "id": "high",
            "type": "adapter",
            "adapter": "fuzzy_art",
            "params": {"rho": 0.7},
        },
    )
    data = add_edge_to_config(
        data,
        {
            "source": "low",
            "target": "high",
            "type": "BOTTOM_UP",
            "transform": {
                "name": "selected_category_to_one_hot",
                "params": {"vector_size": 4},
            },
        },
    )
    assert graph_config_from_dict(json.loads(config_dict_to_json(data))).name
    assert graph_config_from_dict(
        __import__("yaml").safe_load(config_dict_to_yaml(data))
    ).name
    removed = remove_module_from_config(data, "high")
    assert len(removed["modules"]) == 1
    assert removed["edges"] == []
    assert remove_edge_from_config(data, 0)["edges"] == []


def test_graph_overview_helpers():
    graph = build_graph_from_config(
        load_graph_config(CONFIG_DIR / "two_level_fuzzy_art.yaml")
    )
    overview = build_graph_overview(graph)
    assert len(build_module_state_table(graph.get_global_state())) == 2
    assert len(build_edge_table(graph)) == 1
    assert overview["edge_count"] == 1
