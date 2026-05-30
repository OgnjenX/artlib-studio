import streamlit as st

def show_search_history(trace, current_sample_index):
    if trace is None or current_sample_index is None or current_sample_index < 0:
        return

    events = trace.events if hasattr(trace, 'events') else trace
    sample_events = [e for e in events if e.payload.get('sample_index') == current_sample_index]

    if not sample_events:
        return

    st.markdown(f"### Sample {current_sample_index} Search History")

    steps_txt = ""
    step = 1

    for e in sample_events:
        etype = e.type.value
        p = e.payload
        if etype == 'CATEGORY_SELECTED':
            cid = p.get('category_id', '?')
            steps_txt += f"**{step}. Category {cid} selected.**\n"
        elif etype == 'MATCH_TEST':
            ms = p.get('match_score', 'None')
            vig = p.get('vigilance', 'None')
            passed = p.get('passed', False)
            if passed:
                op = ">="
            else:
                op = "<"
            if ms != 'None' and vig != 'None':
                steps_txt += f"Match = {float(ms):.4f} {op} vigilance = {float(vig):.4f}.\n"
        elif etype == 'RESET':
            cid = p.get('category_id', '?')
            steps_txt += f"Reset category {cid}.\n\n"
            step += 1
        elif etype == 'RESONANCE':
            steps_txt += "Resonance achieved.\n"
        elif etype == 'LEARNING':
            steps_txt += "Learning applied.\n\n"
            step += 1
        elif etype == 'CATEGORY_CREATED':
            steps_txt += "No existing category passed vigilance.\nNew category created.\n\n"
            step += 1

    if steps_txt:
        st.markdown(steps_txt)

