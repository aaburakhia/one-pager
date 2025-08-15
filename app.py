# app.py

import streamlit as st
import fitz  # PyMuPDF
import os
import requests

# --- Configuration ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- Core Functions ---

def parse_pdf(file_stream):
    """Extracts text from a PDF file stream, limiting the length for API calls."""
    text = ""
    with fitz.open(stream=file_stream.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    # Truncate text to a reasonable length to fit within API context limits
    return text[:15000]

def call_groq_api(prompt, paper_text):
    """Generic function to call the Groq API with a specific prompt."""
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is not set! Please add it to your Streamlit secrets.")
        return None

    payload = {
        # --- THIS IS THE UPGRADED LINE ---
        "model": "llama3-70b-8192", # Using the more powerful 70B model
        "messages": [
            {"role": "system", "content": "You are an expert research assistant specializing in synthesizing complex academic papers into clear, concise summaries. You are precise and adhere strictly to the user's requested format."},
            {"role": "user", "content": f"{prompt}\n\nHere is the paper's text:\n\n{paper_text}"}
        ]
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        st.error(f"API Error: {response.status_code} - {response.text}")
        return f"Error analyzing this section. Status Code: {response.status_code}"

def generate_the_one_pager(paper_text):
    """Generates the structured summary by calling the LLM for each section."""
    
    st.info("Synthesizing with our top-tier model... this may take a moment.")
    
    with st.spinner("Identifying the core research question..."):
        question = call_groq_api("What is the single, specific research question the authors are trying to answer? State it in one sentence.", paper_text)

    with st.spinner("Extracting the core contribution..."):
        contribution = call_groq_api("What is the main, novel contribution of this work to its field? Describe it in one or two sentences.", paper_text)

    with st.spinner("Detailing the methodology..."):
        methodology = call_groq_api("List the key methodologies, techniques, or data sources used in this paper as a bulleted list.", paper_text)

    with st.spinner("Summarizing the key findings..."):
        findings = call_groq_api("List the 3-5 most important findings or results from the paper as a numbered list.", paper_text)

    with st.spinner("Identifying limitations..."):
        limitations = call_groq_api("What are the key weaknesses, limitations, or open questions mentioned by the authors in the paper? List them as a bulleted list.", paper_text)

    summary = {
        "core_idea": {"question": question, "contribution": contribution},
        "methodology": methodology,
        "key_findings": findings,
        "weaknesses": limitations
    }
    return summary

# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="The One Pager")

st.title("📄 The One Pager")
st.markdown("##### Turn any dense academic paper into a structured, one-page summary.")

if not GROQ_API_KEY:
    st.warning("The Groq API key is not configured. The app will not be able to summarize papers. Please ask the app owner to add the key.", icon="⚠️")

uploaded_file = st.file_uploader("Upload a PDF to begin", type="pdf", label_visibility="collapsed")

if uploaded_file and GROQ_API_KEY:
    paper_text = parse_pdf(uploaded_file)
    summary_data = generate_the_one_pager(paper_text)

    st.header("🔬 The One Pager Summary")
    st.markdown("---")

    st.subheader("💡 Core Idea")
    st.write(f"**Main Research Question:** {summary_data['core_idea']['question']}")
    st.write(f"**Core Contribution:** {summary_data['core_idea']['contribution']}")
    st.markdown("---")

    st.subheader("🛠️ Methodology")
    st.markdown(summary_data["methodology"])
    st.markdown("---")

    st.subheader("📈 Key Findings")
    st.markdown(summary_data["key_findings"])
    st.markdown("---")

    st.subheader("🤔 Limitations & Open Questions")
    st.markdown(summary_data["weaknesses"])
