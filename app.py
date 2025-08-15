# app.py

import streamlit as st
import pdfplumber
import os
import requests
import json

# --- Configuration ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- CSS for the "Paper" Look ---
def local_css():
    st.markdown("""
    <style>
    .paper-container {
        border: 1px solid #e0e0e0; padding: 2rem; border-radius: 10px; background-color: #ffffff;
        font-family: serif; line-height: 1.6;
    }
    .paper-container h2, .paper-container h3 {
        border-bottom: 1px solid #e0e0e0; padding-bottom: 5px;
    }
    .analysis-section {
        background-color: #f8f9fa; padding: 1.2rem; border-radius: 8px; margin-top: 1rem; border-left: 5px solid #0d6efd;
    }
    .section-title { font-weight: bold; margin-bottom: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

# --- Core Functions ---
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

    st.info("Synthesizing your Expert-level analysis... this may take a moment.")

    # --- THE NEW "EXPERT-LEVEL" PROMPT with NESTED JSON ---
    prompt = """
    Analyze the following academic paper text as a world-class expert and peer reviewer.
    Generate a deeply structured analysis in a single JSON object.

    The JSON object must contain the following top-level keys: "summary", "analysis", "synthesis".

    1. "summary": A concise overview of the paper.
       - "question": A single sentence stating the main research question.
       - "contribution": One or two sentences describing the paper's core contribution.
       - "findings": A markdown numbered list of the 3-5 most important results.

    2. "analysis": A critical evaluation of the paper's components.
       - "methodology": A markdown bulleted list of the key methods.
       - "stated_limitations": A markdown bulleted list of limitations explicitly mentioned by the authors.
       - "critique": Your own critical analysis. This is a nested object with keys:
         - "methodological_flaws": Potential unstated weaknesses in the methodology or experimental design.
         - "interpretive_flaws": Alternative interpretations of the results that the authors may have missed.
         - "generalizability": An assessment of how well these findings might apply to other contexts.

    3. "synthesis": Placing the paper in a broader scientific context.
       - "impact": A nested object with keys:
         - "short_term": The potential impact on the immediate field in the next 1-3 years.
         - "long_term": The potential long-term significance or paradigm-shifting impact in 5-10 years.
       - "future_work": A proposal for a high-impact follow-up study. This is a nested object with keys:
         - "hypothesis": A clear, testable hypothesis.
         - "proposed_method": A brief description of the experiment to test the hypothesis.
         - "expected_outcome": What would the results of this experiment tell the scientific community?

    Do not include any text outside of the single, valid JSON object.
    """
    
    payload = {"model": "llama3-70b-8192", "messages": [{"role": "system", "content": "You are a world-class academic expert that provides deeply structured analysis in a valid JSON format."}, {"role": "user", "content": f"{prompt}\n\nHere is the paper's text:\n\n{paper_text}"}], "response_format": {"type": "json_object"}}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    with st.spinner("Our AI expert is performing a deep-level analysis..."):
        response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            response_text = response.json()['choices'][0]['message']['content']
            return json.loads(response_text)
        except (json.JSONDecodeError, KeyError) as e:
            st.error(f"Error parsing the AI's response. Details: {e}")
            return None
    else:
        st.error(f"API Error: {response.status_code} - {response.text}")
        return None

# --- THE NEW "EXPERT-LEVEL" UI ---
st.set_page_config(layout="wide", page_title="The One Pager")
st.title("📄 The One Pager")
st.markdown("##### Turn any dense academic paper into an expert-level, structured analysis.")

local_css()

if not GROQ_API_KEY:
    st.warning("The Groq API key is not configured...", icon="⚠️")

uploaded_file = st.file_uploader("Upload a PDF to begin", type="pdf", label_visibility="collapsed")

if uploaded_file and GROQ_API_KEY:
    data = generate_the_one_pager(parse_pdf(uploaded_file))

    if data:
        # Using .get() with default dictionaries {} or strings "" is a robust way to prevent errors
        summary = data.get("summary", {})
        analysis = data.get("analysis", {})
        synthesis = data.get("synthesis", {})

        st.markdown('<div class="paper-container">', unsafe_allow_html=True)
        st.header("The One Pager: Expert Analysis")
        st.markdown("---")
        
        # --- Section 1: Summary ---
        st.subheader("📝 Summary")
        st.write(f"**Main Research Question:** {summary.get('question', 'N/A')}")
        st.write(f"**Core Contribution:** {summary.get('contribution', 'N/A')}")
        st.markdown("<div class='section-title'>Key Findings:</div>", unsafe_allow_html=True)
        st.markdown(summary.get('findings', 'N/A'))
        st.markdown("---")

        # --- Section 2: Analysis ---
        st.subheader("🧐 Critical Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-title'>Methodology:</div>", unsafe_allow_html=True)
            st.markdown(analysis.get('methodology', 'N/A'))
            st.markdown("<div class='section-title'>Stated Limitations:</div>", unsafe_allow_html=True)
            st.markdown(analysis.get('stated_limitations', 'N/A'))
        with col2:
            critique = analysis.get("critique", {})
            st.markdown("<div class='section-title'>Peer Reviewer's Critique:</div>", unsafe_allow_html=True)
            st.markdown(f"**Methodological Flaws:** {critique.get('methodological_flaws', 'N/A')}")
            st.markdown(f"**Interpretive Flaws:** {critique.get('interpretive_flaws', 'N/A')}")
            st.markdown(f"**Generalizability:** {critique.get('generalizability', 'N/A')}")
        st.markdown("---")

        # --- Section 3: Synthesis ---
        st.subheader("🌍 Broader Context & Synthesis")
        impact = synthesis.get("impact", {})
        future_work = synthesis.get("future_work", {})

        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Significance & Impact:</div>", unsafe_allow_html=True)
        st.markdown(f"**Short-term (1-3 Years):** {impact.get('short_term', 'N/A')}")
        st.markdown(f"**Long-term (5-10 Years):** {impact.get('long_term', 'N/A')}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Proposed Future Research:</div>", unsafe_allow_html=True)
        st.markdown(f"**Hypothesis:** {future_work.get('hypothesis', 'N/A')}")
        st.markdown(f"**Proposed Method:** {future_work.get('proposed_method', 'N/A')}")
        st.markdown(f"**Expected Outcome:** {future_work.get('expected_outcome', 'N/A')}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
