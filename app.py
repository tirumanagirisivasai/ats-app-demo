import streamlit as st
from model_load import *
import fitz
from docx import Document
import io
#prevents the reloading of the model on every interaction
@st.cache_resource  
def cached_load_model():
    try:

        load_model()
        return True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return False
    
cached_load_model()

def extract_text_from_file(file_path):
    file_name = file_path.name.lower()
    file_bytes = file_path.read()

    if file_name.endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    
    elif file_name.endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    else:
        raise ValueError("Unsupported file format. Please upload a PDF or DOCX file.")

st.title("Demo ATS Application")
#st.write("This is a demo application for ATS using Streamlit.")

st.markdown("### Job description generator")
with st.container(border=True):
    raw_job_description = st.text_area("Job Description", "Enter the job description here...")
    if st.button("Generate ✨"):

        job_desc = summarize_jd(raw_job_description, max_tokens=512)
    
    if job_desc["Status"] == "OK":
        generated_jd = job_desc["outputs"][0]
        skills_required = job_desc["outputs"][1]
    else:
        generated_jd = "Error generating job description."

    st.markdown(generated_jd)
    st.markdown(skills_required)

st.markdown("### Resume Analysis")
with st.container(border=True):
    
    resume = st.file_uploader("Upload your Resume", type=["pdf", "docx"])
    if resume is not None:
        st.write("Resume uploaded successfully!")
        # Function call to analyze resume against job description
        parsed_resume = extract_text_from_file(resume)
        skills = extract_skills(parsed_resume, max_tokens=512) #output is in JSON format i.e., {"status":"OK", "outputs":{"Skills":[], "E-mail":"", "experience":""}}
        if skills["Status"] == "OK":
            skills = skills["outputs"]
            candidate_skills = skills['skills']
            candidate_exp = skills['experience']
            st.write("Skills ", skills['skills'])
            st.write("E-mail ", skills['E-mail'])
            st.write("Experience ", skills['experience'])
        else:
            st.write("Error extracting skills from resume.")

    else:
        st.write("Invalid file format. Please upload a PDF or DOCX file.")

# Initialize session state
if "s1_value" not in st.session_state:
    st.session_state.s1_value = None

if "s2_value" not in st.session_state:
    st.session_state.s2_value = None


options = ["Job desc", "resume"]




st.markdown("### Match candidates")
with st.container(border=True):
    s1, s2 = st.columns(2)
    
    with s1:
        value = st.selectbox("Job details", ['Select the new JD', "use the generated JD"], placeholder="Select the job description source")
        st.session_state.s1_value = value
    with s2:
        value = st.selectbox("Resume details", ['Select the new Resume', "use the uploaded Resume"], placeholder="Select the resume source")
        st.session_state.s2_value = value

    if st.button("Match Candidates 🔍"):
        if st.session_state.s1_value == 'use the generated JD':
            match_jd = generated_jd
        elif st.session_state.s1_value == "Select the new JD":
            jd_desc = st.text_input("Please enter the job description", key="jd_input")
            if st.button("Submit JD", key="jd_submit"):
                match_jd = summarize_jd(jd_desc, max_tokens=512)
                if match_jd["Status"] == "OK":
                    match_jd = match_jd["outputs"][0]
                    st.write("Job Description generated successfully!")
                    st.markdown(match_jd)
                match_job_skills = match_jd["outputs"][1]
                match_job_experience = match_jd["outputs"][2]
        else:
            st.error("Please select a valid job description source.")
            st.stop()
        
        if st.session_state.s2_value == 'use the uploaded Resume':
            match_candidate_skills = candidate_skills
            match_candidate_exp = candidate_exp
        elif st.session_state.s2_value == "Select the new Resume":
            comp_resume = st.file_uploader("Upload the resume to compare", type=["pdf", "docx"], key="comp_resume_uploader")
            if comp_resume is not None:
                st.write("Resume uploaded successfully!")
                # Function call to analyze resume against job description
                parsed_resume = extract_text_from_file(comp_resume)
                match_skills = extract_skills(parsed_resume, max_tokens=512) #output is in JSON format i.e., {"status":"OK", "outputs":{"Skills":[], "E-mail":"", "experience":""}}
                if match_skills["Status"] == "OK":
                    match_skills = match_skills["outputs"]
                    match_candidate_skills = match_skills['skills']
                    match_candidate_exp = match_skills['experience']
                    st.write("Skills ", match_skills['skills'])
                    st.write("E-mail ", match_skills['E-mail'])
                    st.write("Experience ", match_skills['experience'])
                else:
                    st.write("Error extracting skills from resume.")

            else:
                st.write("Invalid file format. Please upload a PDF or DOCX file.")
        else:
            st.error("Please select a valid resume source.")
            st.stop()
        match_result = skills_matcher(match_job_skills, match_candidate_skills, match_candidate_exp, match_job_experience) #expected to return a score based on the skills matched
        score = score_candidate_v4(match_result, int(match_job_experience), match_job_skills)
        if 90<=score<=100:
            st.write("Perfect Candidate")
        elif 75<=score<90:
            st.write("Strong Candidate")
        elif 60<=score<75:
            st.write("Good Candidate, But not enough to shortlist")
        else:
            st.write("Unfit Candidate")