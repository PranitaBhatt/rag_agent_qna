import streamlit as st
from agent import generate_answer

st.title("🧠 RAG-Powered Q&A Assistant")

query = st.text_input("Ask a question:")
if query:
    with st.spinner("Thinking..."):
        answer = generate_answer(query)
        st.markdown(f"### Answer:\n{answer}")
