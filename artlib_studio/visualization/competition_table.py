import streamlit as st

def show_competition_table(trace, current_sample_index):
    if trace is None or current_sample_index is None or current_sample_index < 0:
        return

    events = trace.events if hasattr(trace, 'events') else trace
    sample_events = [e for e in events if e.payload.get('sample_index') == current_sample_index]

    if not sample_events:
        return

    cats = {}
    for e in sample_events:
        c_id = e.payload.get('category_id')
        if c_id is not None:
            if c_id not in cats:
                cats[c_id] = {'category_id': c_id, 'choice_score': '-', 'match_score': '-', 'vigilance': '-', 'status': '-'}

            val = e.type.value
            if val == 'CATEGORY_EVALUATED' or val == 'CATEGORY_SELECTED':
                cs = e.payload.get('choice_score')
                if cs is not None: cats[c_id]['choice_score'] = f'{cs:.4f}'
                cats[c_id]['status'] = val.split('_')[1].lower()
            elif val == 'MATCH_TEST':
                ms = e.payload.get('match_score')
                vig = e.payload.get('vigilance')
                if ms is not None: cats[c_id]['match_score'] = f'{ms:.4f}'
                if vig is not None: cats[c_id]['vigilance'] = f'{vig:.4f}'
            elif val in ['RESET', 'RESONANCE', 'LEARNING']:
                cats[c_id]['status'] = val.lower()
        elif e.type.value == 'CATEGORY_CREATED':
            c_new = e.payload.get('created_index', -1)
            if c_new != -1:
                if c_new not in cats:
                    cats[c_new] = {'category_id': c_new, 'choice_score': '-', 'match_score': '-', 'vigilance': '-', 'status': 'created'}
                else:
                    cats[c_new]['status'] = 'created'

    if cats:
        import pandas as pd
        st.markdown('### Category Competition')
        df = pd.DataFrame(list(cats.values()))
        st.dataframe(df, hide_index=True)

