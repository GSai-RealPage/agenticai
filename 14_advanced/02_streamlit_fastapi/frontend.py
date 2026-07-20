import streamlit as st
import requests

st.title("Simple LLM Chat")

message = st.text_input("Ask something")

if st.button("Send"):

    response = requests.post(
        "http://127.0.0.1:8000/chat",
        json={
            "message": message
        }
    )

    st.write("### LLM Response")
    st.write(response.json()["reply"])