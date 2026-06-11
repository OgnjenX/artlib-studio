"""Streamlit v2 component wrapper for the ART composition graph canvas."""
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any, Dict, Optional

import streamlit as st


@lru_cache(maxsize=1)
def _component():
    return st.components.v2.component(
        "artlib-studio.artlib_studio_graph_canvas",
        js="index.js",
        css="index.css",
        html='<div class="react-root"></div>',
    )


def art_graph_canvas(
    graph: Dict[str, Any],
    *,
    revision: int,
    runtime: Optional[Dict[str, Any]] = None,
    key: str = "art_graph_canvas",
    on_graph_change: Optional[Callable[[], None]] = None,
):
    """Render an editable ART composition graph."""
    return _component()(
        key=key,
        data={
            "graph": graph,
            "revision": revision,
            "runtime": runtime,
        },
        default={"graph": graph},
        height=620,
        width="stretch",
        on_graph_change=on_graph_change or (lambda: None),
    )
