# ART Composition Protocol

ARTLib Studio needs a composition protocol because a collection of ART
algorithms is not automatically a computational system. Modules need a common
way to receive input, expose category competition and resonance state, exchange
signals, and report what happened.

This milestone creates a software protocol for ART module composition. It does
not claim to fully simulate Grossberg's biological neural dynamics.

## Single Model and Composition Graph

A single ART model receives samples and performs category choice, vigilance
testing, reset search, resonance, and learning. An `ARTCompositionGraph`
coordinates multiple `ComposableARTModule` instances. The graph owns typed
edges, delivers signals, invokes modules, propagates outputs, and records a
composition event log.

`AdapterARTModule` is the initial bridge from existing ARTLib Studio adapters
to this protocol. It currently supports BaseART-style incremental models
through `InstrumentedART`. It supports explicit `rho` modulation for the
Fuzzy ART composition experiments. It does not implement generic model
modulation or learned top-down ART templates.

## Signals

Signals are immutable dataclasses carrying a source module, optional target,
step index, structured payload, and human-readable summary. The initial signal
vocabulary covers:

- external or inter-module input
- category activation and selection
- top-down expectation
- match and vigilance results
- reset and resonance
- learning

Payloads remain dictionaries so model-specific details can evolve without
expanding the base protocol for every ART variant.

## Edges

- **Bottom-up** edges carry evidence or category activity toward another module.
- **Top-down expectation** edges carry an expectation toward a lower or related
  module.
- **Modulatory** edges carry context that can influence processing without
  representing primary input. The current implementation supports explicit
  vigilance (`rho`) changes on adapter modules.
- **Associative** edges connect category or state representations between
  modules.
- **Reset propagation** edges communicate reset state to another module.

An edge may define a callable transform or a descriptive transform name.
Transforms are intentionally lightweight in this milestone.

## Scheduling and Resonance

The discrete scheduler delivers an external batch, steps modules that received
signals, collects their outputs, and propagates those outputs over outgoing
edges. Recurrent graphs are allowed, but each external graph step has a bounded
number of settling iterations. A graph records either `GRAPH_SETTLED` or
`GRAPH_FAILED_TO_SETTLE`.

Composition-level resonance currently means that modules can emit and exchange
resonance signals. It does not yet implement a global biological resonance
equation or synchronized multi-module learning rule.

## Current Limitations

- Edge types describe connection intent but do not yet enforce signal schemas.
- Signal compatibility is validated by transforms and receiving modules rather
  than a complete static type system.
- `AdapterARTModule` handles input signals and BaseART-style incremental
  learning. Top-down expectations are evaluated by the scheduler rather than
  applied to native ART category search.
- Modulation currently supports only `rho`; unsupported parameters fail
  explicitly.
- Streaming inputs for complement-coded adapters must already be normalized to
  the `[0, 1]` range; graph-wide normalization policy is not yet defined.
- Category creation produces a selected-category signal, but the native trace
  has no separate composition category-created signal yet.
- Runtime state is in memory and scheduling is deterministic and sequential.
- The Streamlit composer uses forms and a static preview rather than
  drag-and-drop.
- No complex Grossberg architecture or biological timing model is included.

These constraints keep the protocol small while leaving clear extension points
for typed routing, expectation-aware adapters, and composition visualizations.

## Two-Level ART Pipeline

The first composition experiment connects two incremental Fuzzy ART modules in
a feed-forward graph:

```text
2D sample -> low-level Fuzzy ART -> one-hot category -> high-level Fuzzy ART
```

The low-level module learns categories over normalized 2D samples. Its selected
category is converted to a fixed-width one-hot vector and delivered to the
high-level module as a new `InputSignal`. The high-level module therefore learns
categories over the observed stream of low-level category identities.

The one-hot selected-category signal is an engineering simplification, not a
full ART top-down/bottom-up neural code. It discards the low-level activation
distribution and prototype geometry, and its configured width limits the
number of low-level categories that can be represented.

This experiment is feed-forward only. The high-level module does not send an
expectation back to the low-level module, coordinate vigilance, or participate
in recurrent settling. It is therefore not a Grossberg-style bidirectional
resonance architecture.

Run the experiment and export its graph trace with:

```bash
python examples/two_level_fuzzy_art_pipeline.py
python examples/export_two_level_composition_trace.py
```

## Bidirectional Expectation Prototype

The bidirectional prototype adds a simplified top-down path:

```text
2D input -> low-level category -> high-level category
                ^                       |
                |---- expectation ------|
```

Bottom-up recognition is unchanged: the low-level Fuzzy ART selects a category,
its identity is encoded as a one-hot vector, and the high-level Fuzzy ART
selects a category over that representation.

An `AssociationMemory` records observed pairs of high-level and low-level
category IDs. When a high-level category becomes active, the top-down edge
emits an `ExpectationSignal` containing the low-level categories previously
associated with it. The scheduler compares that expected set with the
low-level module's current category.

If the expected set is empty, the graph records `EXPECTATION_UNAVAILABLE` and
`CROSS_MODULE_EXPECTATION_UNKNOWN`.

If the current category is expected, the graph records
`EXPECTATION_MATCHED` and `CROSS_MODULE_RESONANCE`.

If the expected set is non-empty but the current category is not in it, the
graph records `EXPECTATION_MISMATCHED` and `CROSS_MODULE_MISMATCH`.

Association updates are committed after expectation evaluation, so the first
occurrence of a new high-level / low-level pair is reported as unknown
expectation rather than mismatch.

This association lookup is an explicit educational mechanism. It is not a
learned top-down Fuzzy ART prototype, does not modify low-level vigilance or
category search, and does not implement synchronized neural dynamics.

This prototype demonstrates the idea of cross-module expectation and
resonance, but it is not yet a full recurrent neural ART circuit.

Run it with:

```bash
python examples/bidirectional_expectation_demo.py
python examples/export_bidirectional_trace.py
```

## Composition Studio

The Streamlit application has a `Composition Experiments` mode for inspecting
the two-level, bidirectional, and modulatory examples. It displays module
state, typed edges, signal flow, selected categories, expectation results, and
the graph event timeline.

The graph diagram is intentionally static. Computation remains in the
composition backend, so graph construction, scheduling, tracing, and
modulation can be tested without Streamlit.

Run the studio with:

```bash
uv run streamlit run artlib_studio/apps/streamlit_app.py
```

## YAML and JSON Graph Configuration

`GraphConfig` describes modules, edges, transforms, initial associations, and
the settling limit. `load_graph_config` and `save_graph_config` support YAML
and JSON. `build_graph_from_config` validates the configuration and constructs
an executable `ARTCompositionGraph`.

Currently supported module types are:

- `adapter`, using the `fuzzy_art` adapter
- `context`, using one or more explicit context rules

Currently supported transforms are:

- `selected_category_to_one_hot`
- `selected_category_to_scalar_vector`
- `selected_category_to_activation_vector`
- `high_category_to_expectation`

Invalid module references, adapters, edge types, transforms, transform
parameters, associations, and context rules raise readable `ValueError`s.

Example configurations are in `examples/configs/`. Run one with:

```bash
uv run python examples/run_composition_config.py \
  examples/configs/bidirectional_expectation.yaml
```

## Graph Composer

The Graph Composer is a config-driven visual editor in Streamlit. It can load
built-in examples, edit YAML or JSON, add and remove modules and edges, manage
association entries, validate and run a graph, and download the configuration
or event trace.

It uses Streamlit forms, tables, and a static graph preview. Drag-and-drop is
not required for graph semantics: typed edges and named transforms define how
signals move through the executable graph.

## Modulatory Context Nodes

A `ContextModule` is an explicit control node. Active `ContextRule`s emit
`ModulatorySignal`s over `MODULATORY` edges before input is processed. The
current adapter implementation supports vigilance parameter `rho` with:

- `set`, `add`, and `multiply` modes
- `current_step`, which applies before input processing and restores the
  previous value only after the graph step settles
- `persistent`, which leaves the changed value active

Higher vigilance requires a stricter category match and can produce finer or
more numerous categories. Lower vigilance permits broader matches and can
produce fewer categories.

The scheduler gives context signals priority over external input in each graph
step. An adapter applies all pending modulation before processing input and
keeps the changed value active throughout category search and learning. A
typical current-step trace is:

```text
MODULATION_SENT
MODULATION_RECEIVED
MODULE_PARAMETER_MODULATED
MODULE_RECEIVED_INPUT
MODULE_SELECTED_CATEGORY / MODULE_RESONATED / MODULE_CATEGORY_CREATED
MODULE_PARAMETER_RESTORED
GRAPH_SETTLED
```

The modulation table in the Composition Studio shows the parameter value
before modulation, the value used during processing, and the restored value.

Run the comparison and trace export with:

```bash
uv run python examples/modulatory_vigilance_demo.py
uv run python examples/export_modulatory_trace.py
```

Context nodes are an educational computational control mechanism. They are
not a full neuromodulatory brain model, and the scheduler does not simulate
synchronized neural dynamics.
