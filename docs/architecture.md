# ARTLib Studio Architecture

AdaptiveResonanceLib
↓
ARTLib Studio Adapter
↓
Instrumentation / TraceRecorder
↓
Event Stream
↓
Capability-aware Visualization

## Why adapters?

* different ART models expose different internals
* not every ART model has 2D category boxes
* not every model uses the same data representation
* compound models may require specialized adapters
* capability-based rendering avoids misleading visualizations
