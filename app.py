# app.py

import streamlit as st
import pdfplumber
import os
import requests
import json

# --- Configuration ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- NEW: Custom CSS for the "Paper" Look ---
def local_css():
    st.markdown("""
    <style>
    .paper-container {
        border: 1px solid #e0e0e0;
        padding: 2rem;
        border-radius: 10px;
        background-color: #ffffff;
        font-family: serif; /* More academic font */
        line-height: 1.6;
    }
    .paper-container h2, .paper-container h3 {
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Core Functions (No changes here) ---
def parse_pdf(file_stream):
    text = ""
    try:
        with pdfplumber.open(file_stream) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text[:15000]
    except Exception as e:
        st.error(f"Error reading the PDF file: {e}")
        return ""

def generate_the_one_pager(paper_text):
    if not paper_text:
        st.error("Could not extract any text from the PDF...")
        return None

    st.info("Synthesizing your One Pager... this may take a moment.")

    # --- UPDATED PROMPT: Added "resources" key ---
    prompt = """
    Analyze the following academic paper text and generate a structured summary in JSON format.
    The JSON object must contain the following keys: "question", "contribution", "methodology", "findings", "limitations", "resources".
    - "question": A single sentence stating the main research question.
    - "contribution": One or two sentences describing the paper's core contribution.
    - "methodology": A markdown bulleted list of the key methods and data sources.
    - "findings": A markdown numbered list of the 3-5 most important results.
    - "limitations": A markdown bulleted list of the key limitations mentioned by the authors.
    - "resources": A markdown bulleted list of any links to datasets, code repositories, or external resources mentioned. If none are found, return "No specific resources were mentioned."
    Do not include any text outside of the JSON object.
    """
    
    payload = {"model": "llama3-70b-8192", "messages": [{"role": "system", "content": "You are a research assistant that provides structured summaries in JSON format."}, {"role": "user", "content": f"{prompt}\n\nHere is the paper's text:\n\n{paper_text}"}], "response_format": {"type": "json_object"}}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    with st.spinner("Our AI is reading and analyzing the entire paper..."):
        response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            response_text = response.json()['choices'][0]['message']['content']
            summary_data = json.loads(response_text)
            return {
                "core_idea": {"question": summary_data.get("question", "Not found"), "contribution": summary_data.get("contribution", "Not found")},
                "methodology": summary_data.get("methodology", "Not found"),
                "key_findings": summary_data.get("findings", "Not found"),
                "weaknesses": summary_data.get("limitations", "Not found"),
                "resources": summary_data.get("resources", "Not found") # NEW
            }
        except (json.JSONDecodeError, KeyError) as e:
            st.error(f"Error parsing the AI's response. Details: {e}")
            return None
    else:
        st.error(f"API Error: {response.status_code} - {response.text}")
        return None

# --- UPDATED Streamlit App UI ---
st.set_page_config(layout="wide", page_title="The One Pager")
st.title("📄 The One Pager")
st.markdown("##### Turn any dense academic paper into a structured, one-page summary.")

local_css() # Apply our custom styles

if not GROQ_API_KEY:
    st.warning("The Groq API key is not configured...", icon="⚠️")

uploaded_file = st.file_uploader("Upload a PDF to begin", type="pdf", label_visibility="collapsed")

if uploaded_file and GROQ_API_KEY:
    summary_data = generate_the_one_pager(parse_pdf(uploaded_file))

    if summary_data:
        st.markdown('<div class="paper-container">', unsafe_allow_html=True) # Start of our custom container
        
        st.header("The One Pager Summary")
        st.markdown("---")
        
        # Core Idea spans the full width, like an abstract
        st.subheader("💡 Core Idea")
        st.write(f"**Main Research Question:** {summary_data['core_idea']['question']}")
        st.write(f"**Core Contribution:** {summary_data['core_idea']['contribution']}")
        st.markdown("---")

        # --- NEW: Two-column layout ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🛠️ Methodology")
            st.markdown(summary_data["methodology"])
            st.markdown("---")
            st.subheader("📈 Key Findings")
            st.markdown(summary_data["key_findings"])
        
        with col2:
            st.subheader("🤔 Limitations & Open Questions")
            st.markdown(summary_data["weaknesses"])
            st.markdown("---")
            # --- NEW: Links & Datasets Section ---
            st.subheader("🔗 Links & Datasets")
            st.markdown(summary_data["resources"])

        st.markdown('</div>', unsafe_allow_html=True) # End of our custom container
