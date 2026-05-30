def build_timeline_state(X, trace, current_event_index, final_labels=None, final_model=None):
    if trace is None:
        return None

    events = trace.events if hasattr(trace, 'events') else trace
    if not events:
        return None

    if current_event_index < 0 or current_event_index >= len(events):
        current_event = None
    else:
        current_event = events[current_event_index]

    processed_samples = set()
    categories_existing = set()
    historical_weights = {}

    current_sample_index = None
    current_category_id = None
    reset_categories = set()
    resonant_category = None
    learned_category = None
    newly_created_category = None

    for i in range(current_event_index + 1):
        ev = events[i]
        payload = ev.payload
        s_idx = payload.get("sample_index")
        if s_idx is not None:
            processed_samples.add(s_idx)
            current_sample_index = s_idx

        c_id = payload.get("category_id")
        created_idx = payload.get("created_index")

        etype = ev.type.value
        if etype == "CATEGORY_CREATED":
            idx = created_idx if created_idx is not None else c_id
            if idx is not None:
                categories_existing.add(idx)
                if "weights_after" in payload:
                    historical_weights[idx] = payload.get("weights_after")
        elif etype == "LEARNING":
            if c_id is not None:
                if "weights_after" in payload:
                    historical_weights[c_id] = payload.get("weights_after")

        if i == current_event_index:
            current_category_id = c_id
            if etype == "RESET":
                resets = payload.get("reset_categories", [])
                if not isinstance(resets, list):
                    resets = [c_id] if c_id is not None else []
                reset_categories = set(resets)
            elif etype == "RESONANCE":
                resonant_category = c_id
            elif etype == "LEARNING":
                learned_category = c_id
            elif etype == "CATEGORY_CREATED":
                newly_created_category = created_idx if created_idx is not None else c_id

    if current_sample_index is not None and current_event is not None and current_event.type.value != "RESET":
        for i in range(current_event_index - 1, -1, -1):
            if events[i].payload.get("sample_index") != current_sample_index:
                break
            if events[i].type.value == "RESET":
                for r in events[i].payload.get("reset_categories", []):
                    reset_categories.add(r)
                if events[i].payload.get("category_id") is not None:
                    reset_categories.add(events[i].payload.get("category_id"))

    total_samples = len(X) if X is not None else 0
    all_indices = set(range(total_samples))
    future_samples = all_indices - processed_samples
    if current_sample_index is not None and current_sample_index in future_samples:
        future_samples.remove(current_sample_index)

    return {
        "current_event": current_event,
        "current_sample_index": current_sample_index,
        "processed_sample_indices": list(processed_samples),
        "future_sample_indices": list(future_samples),
        "historical_weights": historical_weights,
        "categories_existing_so_far": list(categories_existing),
        "current_category_id": current_category_id,
        "reset_categories_for_current_sample": list(reset_categories),
        "resonant_category_for_current_sample": resonant_category,
        "learned_category_for_current_sample": learned_category,
        "newly_created_category_id": newly_created_category
    }
