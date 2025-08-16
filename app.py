# app.py

import streamlit as st
import pdfplumber
import os
import requests
import json
from typing import Dict, List, Any

# --- Configuration ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- PART 1: OUR PROVEN BACKEND LOGIC ---

def parse_pdf(file_stream):
    """Extracts text from a PDF file stream using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(file_stream) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text[:15000] # Limit context length for API
    except Exception as e:
        st.error(f"Error reading the PDF file: {e}")
        return None

def get_expert_analysis(paper_text: str) -> Dict[str, Any]:
    """Calls the Groq API with the expert-level prompt."""
    if not paper_text:
        return None
    
    # The expert-level prompt from the other AI
    prompt = """
    You are an expert academic reviewer analyzing a research paper. Provide a comprehensive analysis in the following JSON structure. Be specific, critical, and insightful in your analysis.
    {
      "metadata": { "title": "The paper's title", "authors": "Primary authors (first 3 if many)", "venue": "Journal/conference if mentioned", "year": "Publication year if available", "field": "Primary research field/discipline" },
      "summary": { "research_question": "The main research question in one clear sentence", "hypothesis": "The central hypothesis or claim being tested (if applicable)", "contribution": "The paper's novel contribution in 1-2 sentences", "key_findings": ["Most significant finding", "Second most important result", "Additional important findings (2-3 more)"] },
      "methodology": { "approach": "High-level methodological approach", "methods": ["Specific method 1", "Specific method 2"], "data": { "type": "Type of data used", "size": "Sample size or data scale", "source": "Where the data came from" }, "strengths": ["Strong aspect of methodology"] },
      "critical_analysis": { "limitations": { "stated": ["Limitation explicitly mentioned by authors"], "unstated": ["Potential limitation not discussed by authors"] }, "methodological_concerns": ["Specific concern about experimental design or analysis"], "alternative_interpretations": ["Different way to interpret the main findings"], "reproducibility": "Assessment of how reproducible this work is (High/Medium/Low) and why" },
      "significance": { "novelty_score": "Rate novelty 1-5 (5=highly novel) with brief justification", "theoretical_impact": "How this advances theoretical understanding", "practical_impact": "Real-world applications or implications", "field_impact": { "immediate": "Impact on field in next 1-2 years", "long_term": "Potential impact in 5-10 years" } },
      "future_directions": { "direct_extensions": ["Natural next step building directly on this work"], "creative_applications": ["Creative application to different domain"], "open_questions": ["Important question this work raises but doesn't answer"] },
      "resources": { "data_availability": "Whether data is available (Yes/No/Partial) and where", "code_availability": "Whether code is available (Yes/No/Partial) and where", "supplementary_materials": ["Link or description of additional resources if mentioned"] },
      "overall_assessment": { "strengths": ["Major strength of the paper"], "weaknesses": ["Main weakness or concern"], "recommendation": "Overall assessment: Accept/Minor Revisions/Major Revisions/Reject with 1-sentence rationale" }
    }
    """
    
    payload = {"model": "llama3-70b-8192", "messages": [{"role": "system", "content": "You are a world-class academic expert providing analysis in a valid JSON format."}, {"role": "user", "content": f"{prompt}\n\nHere is the paper's text:\n\n{paper_text}"}], "response_format": {"type": "json_object"}}
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            return json.loads(response.json()['choices'][0]['message']['content'])
        except (json.JSONDecodeError, KeyError) as e:
            st.error(f"Error parsing the AI's response. Details: {e}")
            return None
    else:
        st.error(f"API Error: {response.status_code} - {response.text}")
        return None

# --- PART 2: THE NEW, SUPERIOR UI FROM THE OTHER AI ---

def load_custom_css():
    st.markdown("""
    <style>
    .main-header { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; text-align: center; }
    .metric-card { background: #f8f9fa; border-left: 4px solid #667eea; padding: 1rem; margin: 0.5rem 0; border-radius: 5px; }
    .strength-card { background: #d4edda; border-left: 4px solid #28a745; padding: 1rem; margin: 0.5rem 0; border-radius: 5px; }
    .weakness-card { background: #f8d7da; border-left: 4px solid #dc3545; padding: 1rem; margin: 0.5rem 0; border-radius: 5px; }
    .finding-item { background: #fff3cd; border-left: 3px solid #ffc107; padding: 0.8rem; margin: 0.3rem 0; border-radius: 3px; }
    .methodology-item { background: #e2e3e5; border-left: 3px solid #6c757d; padding: 0.8rem; margin: 0.3rem 0; border-radius: 3px; }
    .score-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 1rem; font-weight: bold; color: white; }
    .score-high { background-color: #28a745; }
    .score-medium { background-color: #ffc107; color: black; }
    .score-low { background-color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)

def create_score_badge(score: str) -> str:
    if not score: return ""
    score_word = score.split()[0].lower()
    if score_word in ['5', 'high', 'accept']: class_name = 'score-high'
    elif score_word in ['3', '4', 'medium', 'minor']: class_name = 'score-medium'
    else: class_name = 'score-low'
    return f'<span class="score-badge {class_name}">{score}</span>'

def format_list_items(items: List[str], card_class: str = "finding-item") -> str:
    if not items: return "<p><em>None specified</em></p>"
    return "".join(f'<div class="{card_class}">{item}</div>' for item in items)

def display_metadata_section(metadata: Dict[str, Any]):
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown(f'<div class="metric-card"><strong>Title:</strong><br>{metadata.get("title", "N/A")}</div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="metric-card"><strong>Authors:</strong><br>{metadata.get("authors", "N/A")}<br><strong>Year:</strong> {metadata.get("year", "N/A")}</div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="metric-card"><strong>Venue:</strong><br>{metadata.get("venue", "N/A")}<br><strong>Field:</strong> {metadata.get("field", "N/A")}</div>', unsafe_allow_html=True)

def display_summary_section(summary: Dict[str, Any]):
    st.header("📋 Research Summary")
    col1, col2 = st.columns(2)
    with col1: st.subheader("🎯 Research Question"); st.info(summary.get('research_question', 'N/A'))
    with col2: st.subheader("💡 Hypothesis"); st.info(summary.get('hypothesis', 'N/A') or "No specific hypothesis stated")
    st.subheader("🚀 Key Contribution"); st.success(summary.get('contribution', 'N/A'))
    st.subheader("🔍 Key Findings"); st.markdown(format_list_items(summary.get('key_findings', []), "finding-item"), unsafe_allow_html=True)

def display_methodology_section(methodology: Dict[str, Any]):
    st.header("🔬 Methodology")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Approach & Methods")
        st.write(f"**Overall Approach:** {methodology.get('approach', 'N/A')}")
        st.markdown("**Specific Methods:**", unsafe_allow_html=True)
        st.markdown(format_list_items(methodology.get('methods', []), "methodology-item"), unsafe_allow_html=True)
    with col2:
        st.subheader("📈 Data Information")
        data = methodology.get('data', {})
        st.write(f"**Type:** {data.get('type', 'N/A')}")
        st.write(f"**Size:** {data.get('size', 'N/A')}")
        st.write(f"**Source:** {data.get('source', 'N/A')}")
        st.subheader("✅ Methodological Strengths"); st.markdown(format_list_items(methodology.get('strengths', []), "strength-card"), unsafe_allow_html=True)

def display_critical_analysis_section(analysis: Dict[str, Any]):
    st.header("🤔 Critical Analysis")
    limitations = analysis.get('limitations', {})
    col1, col2 = st.columns(2)
    with col1: st.subheader("⚠️ Stated Limitations"); st.markdown(format_list_items(limitations.get('stated', []), "methodology-item"), unsafe_allow_html=True)
    with col2: st.subheader("🚨 Potential Unstated Issues"); st.markdown(format_list_items(limitations.get('unstated', []), "weakness-card"), unsafe_allow_html=True)
    st.subheader("🤔 Methodological Concerns"); st.markdown(format_list_items(analysis.get('methodological_concerns', []), "weakness-card"), unsafe_allow_html=True)
    st.subheader("🔄 Alternative Interpretations"); st.markdown(format_list_items(analysis.get('alternative_interpretations', []), "methodology-item"), unsafe_allow_html=True)
    st.subheader("🔄 Reproducibility Assessment"); st.markdown(f'<div class="metric-card">{analysis.get("reproducibility", "N/A")}</div>', unsafe_allow_html=True)

def display_significance_section(significance: Dict[str, Any]):
    st.header("⭐ Significance & Impact")
    col1, col2, col3 = st.columns(3)
    with col1:
        novelty = significance.get('novelty_score', 'N/A')
        st.subheader("🆕 Novelty Score")
        st.markdown(f'{create_score_badge(novelty)}<br><small>{novelty}</small>', unsafe_allow_html=True)
    with col2: st.subheader("🧠 Theoretical Impact"); st.info(significance.get('theoretical_impact', 'N/A'))
    with col3: st.subheader("🌍 Practical Impact"); st.info(significance.get('practical_impact', 'N/A'))
    field_impact = significance.get('field_impact', {})
    st.subheader("📅 Timeline of Impact")
    col1, col2 = st.columns(2)
    with col1: st.write("**Immediate (1-2 years):**"); st.success(field_impact.get('immediate', 'N/A'))
    with col2: st.write("**Long-term (5-10 years):**"); st.success(field_impact.get('long_term', 'N/A'))

def display_future_directions_section(future: Dict[str, Any]):
    st.header("🔮 Future Research Directions")
    tabs = st.tabs(["Direct Extensions", "Creative Applications", "Open Questions"])
    with tabs[0]: st.markdown(format_list_items(future.get('direct_extensions', []), "strength-card"), unsafe_allow_html=True)
    with tabs[1]: st.markdown(format_list_items(future.get('creative_applications', []), "finding-item"), unsafe_allow_html=True)
    with tabs[2]: st.markdown(format_list_items(future.get('open_questions', []), "methodology-item"), unsafe_allow_html=True)

def display_resources_section(resources: Dict[str, Any]):
    st.header("📚 Resources & Availability")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Data Available", resources.get('data_availability', 'N/A'))
    with col2: st.metric("Code Available", resources.get('code_availability', 'N/A'))
    with col3: st.metric("Supplementary Materials", len(resources.get('supplementary_materials', [])))
    if resources.get('supplementary_materials'):
        st.subheader("🔗 Additional Resources")
        for material in resources.get('supplementary_materials'): st.write(f"- {material}")

def display_overall_assessment_section(assessment: Dict[str, Any]):
    st.header("📊 Overall Assessment")
    col1, col2 = st.columns(2)
    with col1: st.subheader("✅ Strengths"); st.markdown(format_list_items(assessment.get('strengths', []), "strength-card"), unsafe_allow_html=True)
    with col2: st.subheader("⚠️ Weaknesses"); st.markdown(format_list_items(assessment.get('weaknesses', []), "weakness-card"), unsafe_allow_html=True)
    st.subheader("🎯 Final Recommendation")
    recommendation = assessment.get('recommendation', 'N/A')
    st.markdown(f'<div class="metric-card">{create_score_badge(recommendation)}<br><br>{recommendation}</div>', unsafe_allow_html=True)

# --- PART 3: THE MAIN APP LOGIC THAT TIES EVERYTHING TOGETHER ---

def main():
    st.set_page_config(page_title="The One Pager: Expert Review", layout="wide")
    load_custom_css()

    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None

    st.markdown('<div class="main-header"><h1>The One Pager: Expert Review</h1></div>', unsafe_allow_html=True)
    
    if not GROQ_API_KEY:
        st.warning("Groq API key is not configured. Please add it to your Streamlit secrets.", icon="⚠️")
        return

    uploaded_file = st.file_uploader("Upload a PDF paper for expert analysis", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        # Use a button to trigger the analysis
        if st.button("Analyze Paper", type="primary"):
            with st.spinner("Parsing PDF and performing expert analysis... This may take up to a minute."):
                paper_text = parse_pdf(uploaded_file)
                if paper_text:
                    st.session_state.analysis_result = get_expert_analysis(paper_text)
                else:
                    # Clear previous results if parsing fails
                    st.session_state.analysis_result = None
    
    # Display the result if it exists in the session state
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        display_metadata_section(result.get('metadata', {}))
        display_summary_section(result.get('summary', {}))
        display_methodology_section(result.get('methodology', {}))
        display_critical_analysis_section(result.get('critical_analysis', {}))
        display_significance_section(result.get('significance', {}))
        display_future_directions_section(result.get('future_directions', {}))
        display_resources_section(result.get('resources', {}))
        display_overall_assessment_section(result.get('overall_assessment', {}))

if __name__ == "__main__":
    main()
