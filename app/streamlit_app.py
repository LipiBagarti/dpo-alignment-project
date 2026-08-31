"""
app/streamlit_app.py
Side-by-side comparison UI: base model vs DPO fine-tuned model.
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="DPO Preference Alignment", layout="wide")
st.title("DPO Preference Alignment — Base vs DPO")
st.caption("Comparing Qwen2.5-1.5B-Instruct before and after DPO fine-tuning on coding explanations.")

prompt = st.text_area("Enter your prompt...", height=100, placeholder="e.g. How do I reverse a list in Python?")

if st.button("Generate", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating responses from both models..."):
            try:
                response = requests.post(f"{API_URL}/generate", json={"prompt": prompt}, timeout=120)
                response.raise_for_status()
                data = response.json()

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("BASE MODEL")
                    st.caption(f"Qwen2.5-1.5B-Instruct · {data['base_generation_time']}s")
                    st.write(data["base_response"])

                with col2:
                    st.subheader("DPO MODEL")
                    st.caption(f"DPO fine-tuned · {data['dpo_generation_time']}s")
                    st.write(data["dpo_response"])

                st.divider()
                vote = st.radio("Which response do you prefer?", ["Base", "DPO", "Tie"], horizontal=True)
                if vote:
                    st.success(f"Recorded preference: {vote}")

            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the API. Is it running? ({e})")

st.divider()
st.caption("Backend: FastAPI · Fine-tuning: DPO + LoRA on Qwen2.5-1.5B-Instruct")
