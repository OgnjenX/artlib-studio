def get_explanation(event):
    if not event:
        return "No event selected."

    etype = event.type.value
    p = event.payload

    if etype == 'INPUT_RECEIVED':
        s_idx = p.get('sample_index', '?')
        return f"ART has now received sample {s_idx}. Only samples 0–{s_idx} have been seen so far. It will compare this pattern against existing expectations."
    elif etype == 'CATEGORY_EVALUATED':
        cid = p.get('category_id', '?')
        cs = p.get('choice_score', 'NaN')
        return f"Category {cid} is evaluated. Choice score is {cs}."
    elif etype == 'CATEGORY_SELECTED':
        cid = p.get('category_id', '?')
        return f"Category {cid} currently has the strongest choice score, so ART tests whether it matches the input well enough."
    elif etype == 'MATCH_TEST':
        passed = p.get('passed', False)
        if passed:
            return "The selected category matches the input strongly enough (match >= vigilance)."
        else:
            return "The selected category does not match the input strongly enough. Because match < vigilance, ART resets this category and continues searching."
    elif etype == 'RESET':
        return "This category is temporarily rejected for the current input. ART will look for another category."
    elif etype == 'RESONANCE':
        return "The input and selected category expectation match sufficiently. ART enters resonance and learning can occur."
    elif etype == 'LEARNING':
        return "This update changes the selected category expectation at this point in the learning timeline."
    elif etype == 'CATEGORY_CREATED':
        return "This category did not exist before this sample. ART created it because no existing category passed vigilance."

    return p.get('explanation', '')
