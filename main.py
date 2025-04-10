import streamlit as st

st.title("📊 Sentiment Analyzer")

text = st.text_area("Enter your sentence:")

if st.button("Analyze"):
    if "good" in text.lower():
        st.success("Positive sentiment 😊")
    elif "bad" in text.lower():
        st.error("Negative sentiment 😞")
    else:
        st.info("Neutral sentiment 😐")
