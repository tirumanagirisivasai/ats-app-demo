import streamlit as st
from model_load import load_model, summarize_jd, extract_skills, skills_matcher, score_candidate_v4
import fitz  # PyMuPDF
from docx import Document
import regex as re
import io
import streamlit as st

# THIS MUST BE FIRST!
st.set_page_config(
    page_title="ATS Resume Matcher",
    page_icon="Briefcase",
    layout="wide",
    initial_sidebar_state="auto"
)


@st.cache_resource(show_spinner="Warming up Mistral-7B model...")
def get_model():
    return load_model()

tokenizer, model = get_model()


# =============================================
# 2. Helper: Extract Text from PDF/DOCX
# =============================================
def extract_text_from_file(file):
    file_bytes = file.read()
    file_name = file.name.lower()

    if file_name.endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    elif file_name.endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    else:
        raise ValueError("Only PDF and DOCX files are supported.")

# =============================================
# 3. Initialize Session State
# =============================================
if "generated_jd" not in st.session_state:
    st.session_state.generated_jd = None
    st.session_state.generated_skills = None
    st.session_state.generated_exp = "0"

if "candidate_skills" not in st.session_state:
    st.session_state.candidate_skills = None
    st.session_state.candidate_exp = "0"
    st.session_state.candidate_email = ""

# =============================================
# 4. UI Starts Here
# =============================================

st.title("ATS Resume Matcher with Mistral-7B")
st.markdown("### Extract skills, generate clean JDs, and score candidates accurately")

# =============================================
# SECTION 1: Job Description Generator
# =============================================
st.markdown("### Job Description Generator")
with st.container(border=True):
    raw_jd = st.text_area("Paste any messy job description", height=200, key="raw_jd_input")
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Generate Clean JD", type="primary", use_container_width=True):
            with st.spinner("Generating structured JD..."):
                result = summarize_jd(raw_jd)
                if result["Status"] == "OK":
                    full_jd, skills, exp = result["output"]
                    st.session_state.generated_jd = full_jd
                    st.session_state.generated_skills = skills
                    st.session_state.generated_exp = exp
                    st.success("Job Description Generated!")
                else:
                    st.error(f"Error: {result['output']}")

    if st.session_state.generated_jd:
        st.markdown(st.session_state.generated_jd)
        st.code(f"Required Skills: {', '.join(st.session_state.generated_skills)}", language="text")
        st.caption(f"Required Experience: {st.session_state.generated_exp} years")

# =============================================
# SECTION 2: Resume Parser
# =============================================
st.markdown("### Resume Skill & Experience Extractor")
with st.container(border=True):
    uploaded_resume = st.file_uploader("Upload Candidate Resume (PDF/DOCX)", type=["pdf", "docx"])

    if uploaded_resume and st.button("Extract Skills from Resume", type="primary"):
        with st.spinner("Analyzing resume..."):
            try:
                resume_text = extract_text_from_file(uploaded_resume)
                result = extract_skills(resume_text)

                if result["Status"] == "OK":
                    data = result["output"]
                    st.session_state.candidate_skills = data.get("skills", [])
                    st.session_state.candidate_exp = data.get("experience", "0")
                    st.session_state.candidate_email = data.get("E-mail", "")

                    st.success("Resume parsed successfully!")
                    st.write("**Email**:", st.session_state.candidate_email or "Not found")
                    st.write("**Experience**:", f"{st.session_state.candidate_exp} years")
                    st.write("**Skills Found**:", ", ".join(st.session_state.candidate_skills) or "None")
                else:
                    st.error("Failed to extract skills. Try a cleaner resume.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

# =============================================
# SECTION 3: Match & Score
# =============================================
st.markdown("### Match Candidate Against Job")
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        jd_source = st.radio("Job Description Source", ["Use Generated JD", "Enter New JD"])
    with col2:
        resume_source = st.radio("Resume Source", ["Use Parsed Resume", "Upload New Resume"])

    # Handle JD input
    job_skills = None
    job_exp = "0"
    if jd_source == "Use Generated JD":
        if st.session_state.generated_skills:
            job_skills = st.session_state.generated_skills
            job_exp = st.session_state.generated_exp
        else:
            st.warning("No generated JD yet. Generate one first!")
    else:
        manual_jd = st.text_area("Enter Job Description for Matching", height=150)
        if st.button("Parse This JD"):
            with st.spinner("Parsing JD..."):
                res = summarize_jd(manual_jd)
                if res["Status"] == "OK":
                    _, skills, exp = res["output"]
                    job_skills = skills
                    job_exp = exp
                    st.success("JD parsed!")

    # Handle Resume input
    candidate_skills_final = None
    candidate_exp_final = "0"
    if resume_source == "Use Parsed Resume":
        if st.session_state.candidate_skills:
            candidate_skills_final = st.session_state.candidate_skills
            candidate_exp_final = st.session_state.candidate_exp
        else:
            st.warning("No resume parsed yet!")
    else:
        new_resume = st.file_uploader("Upload new resume for comparison", type=["pdf", "docx"], key="match_resume")
        if new_resume and st.button("Parse New Resume"):
            with st.spinner("Parsing..."):
                text = extract_text_from_file(new_resume)
                res = extract_skills(text)
                if res["Status"] == "OK":
                    data = res["output"]
                    candidate_skills_final = data.get("skills", [])
                    candidate_exp_final = data.get("experience", "0")
                    st.success("Resume parsed!")

    # FINAL MATCH BUTTON
    if st.button("Run ATS Match & Score", type="primary", use_container_width=True):
        if not job_skills or not candidate_skills_final:
            st.error("Both Job Description and Resume must be loaded!")
        else:
            with st.spinner("Running ATS matching engine..."):
                match_data = skills_matcher(
                    js=job_skills,
                    cs=candidate_skills_final,
                    cex=candidate_exp_final,
                    jex=job_exp
                )

                required_exp_int = int(job_exp.split()[0]) if job_exp.replace("-", "").isdigit() else int(job_exp or 0)
                score = score_candidate_v4(match_data, required_exp_int, critical_skills=job_skills[:5])  # top 5 = critical

                st.markdown(f"## ATS Score: **{score}/100**")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Exact Matches", len(match_data.get("exact_matches", [])))
                with col2:
                    st.metric("Semantic Matches", len(match_data.get("semantic_matches", [])))
                with col3:
                    st.metric("Partial Matches", len(match_data.get("partial_matches", [])))
                with col4:
                    st.metric("Missing Skills", len(match_data.get("missing_skills", [])))

                if score >= 90:
                    st.balloons()
                    st.success("Perfect Match – Strong Hire!")
                elif score >= 75:
                    st.success("Strong Candidate – Shortlist!")
                elif score >= 60:
                    st.warning("Good but Needs Review")
                else:
                    st.error("Not a Strong Fit")

                with st.expander("Detailed Match Breakdown"):
                    st.json(match_data, expanded=False)


# Rest of your beautiful app below...
st.title("Advanced ATS Resume Matcher")
st.markdown("Powered by **Mistral-7B-Instruct** • Local • No API keys")