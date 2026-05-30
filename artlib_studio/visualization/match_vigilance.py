import streamlit as st

def show_match_vigilance_panel(current_event):
    if not current_event:
        return

    payload = current_event.payload
    match_score = payload.get("match_score", None)
    vigilance = payload.get("vigilance", None)

    if match_score is None or vigilance is None:
        st.write("Match/vigilance comparison is not available for this event.")
        return

    passed = payload.get("passed", match_score >= vigilance)
    if "reset_categories" in payload:
        passed = False
    elif current_event.type.value in ["RESONANCE", "LEARNING"]:
        passed = True

    result_text = "PASSED (RESONANCE)" if passed else "FAILED (RESET)"
    color = "green" if passed else "red"

    st.markdown("### Match vs Vigilance")

    col1, col2, col3 = st.columns(3)
    col1.metric("Match Score", f"{match_score:.4f}")
    col2.metric("Vigilance", f"{vigilance:.4f}")

    st.markdown(f"**Result**: :{color}[{result_text}]")

    max_val = max(1.0, float(match_score), float(vigilance))
    if max_val == 0:
        max_val = 1.0
    m_pct = int(match_score / max_val * 100)
    v_pct = int(vigilance / max_val * 100)

    html = f"""
    <div style='margin-top: 10px;'>
        <div style='display: flex; align-items: center; margin-bottom: 5px;'>
            <div style='width: 80px;'>Match</div>
            <div style='flex-grow: 1; background: #e0e0e0; height: 20px; border-radius: 3px;'>
                <div style='width: {m_pct}%; background: {color}; height: 100%; border-radius: 3px;'></div>
            </div>
        </div>
        <div style='display: flex; align-items: center;'>
            <div style='width: 80px;'>Vigilance</div>
            <div style='flex-grow: 1; background: #e0e0e0; height: 20px; border-radius: 3px;'>
                <div style='width: {v_pct}%; background: #333333; height: 100%; border-radius: 3px;'></div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

