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
through `InstrumentedART`. It does not yet implement generic expectation or
modulation handling, and it does not yet make a full two-level ART pipeline.

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
  representing primary input.
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
- The scheduler propagates module outputs over outgoing edges without
  edge-specific gating.
- `AdapterARTModule` handles input signals and BaseART-style incremental
  learning; expectation and modulation are not yet applied to model internals.
- Streaming inputs for complement-coded adapters must already be normalized to
  the `[0, 1]` range; graph-wide normalization policy is not yet defined.
- Category creation produces a selected-category signal, but the native trace
  has no separate composition category-created signal yet.
- Runtime state is in memory and scheduling is deterministic and sequential.
- No graphical editor, complex Grossberg architecture, or biological timing
  model is included.

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
