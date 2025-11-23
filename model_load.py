import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM


global model, tokenizer
local_model_dir = "./mistral_model"

def load_model():
    if os.path.exists(local_model_dir):
        if os.listdir(local_model_dir) == []:
            print("Model files not found in the directory, downloading...")

            try:
                model_name = "mistralai/Mistral-7B-Instruct-v0.3"

                # Download and save locally
                tokenizer_download = AutoTokenizer.from_pretrained(model_name)
                tokenizer_download.save_pretrained(local_model_dir)

                model_download = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype="auto"
                )

                model_download.save_pretrained(local_model_dir)
                return {"Status":"Model Downloaded into the " + local_model_dir, "model_path":local_model_dir}

            except Exception as e:
                return {"Status":"Error While Downloading Model"}
        else:
            print("Model files found in the directory, loading...")
    else:
        os.makedirs(local_model_dir)
        print("Model directory created, downloading model...")

        try:
            model_name = "mistralai/Mistral-7B-Instruct-v0.3"

            # Download and save locally
            tokenizer_download = AutoTokenizer.from_pretrained(model_name)
            tokenizer_download.save_pretrained(local_model_dir)

            model_download = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto"
            )

            model_download.save_pretrained(local_model_dir)
            return {"Status":"Model Downloaded into the " + local_model_dir, "model_path":local_model_dir}

        except Exception as e:
            return {"Status":"Error While Downloading Model"+str(e)}    
    
    try:
        global tokenizer, model
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            local_model_dir,
            device_map="auto",
            torch_dtype=torch.float16
            ).eval()
        
        return {"Status":"Model Loaded Successfully", "model_path":local_model_dir}
    except Exception as e:
        return {"Status":"Error While Loading Model"+str(e)}    

def score_candidate_v4(data:dict, required_years:int, critical_skills=None):
    if critical_skills is None:
        critical_skills = []

    # --- 1. SKILL SCORING (Max 60 pts) ---
    # We create a bucket. Exact fills it fast; Semantic fills it slower.
    # We stop adding points once we hit 60 to prevent "keyword stuffing" inflation.
    
    skill_pts = 0
    skill_pts += len(data.get("exact_matches", [])) * 10
    skill_pts += len(data.get("semantic_matches", [])) * 6  # Lowered from 8 to 6
    skill_pts += len(data.get("partial_matches", [])) * 2   # Lowered from 3 to 2
    
    # Hard cap on positive skill points
    skill_score = min(skill_pts, 60)

    # --- 2. EXPERIENCE SCORING (Max 40 pts) ---
    # Adjusted to sum to 40 so 60+40=100
    actual_years = data.get("candidate_relevant_years", 0)
    gap = actual_years - required_years

    if gap >= 3:
        exp_pts = 40       # Bonus for solid seniority
    elif gap >= 1:
        exp_pts = 35       # Meets + buffer
    elif gap >= 0:
        exp_pts = 30       # Meets exactly
    elif gap >= -2:
        exp_pts = 15       # Slightly under (Seniority Penalty)
    else:
        exp_pts = 0        # Significantly under

    # --- 3. PENALTIES (Subtractive) ---
    missing = data.get("missing_skills", [])
    
    # Critical misses hit hard
    crit_miss_count = sum(1 for skill in missing if skill in critical_skills)
    penalty_critical = crit_miss_count * 25 
    
    # Non-critical misses shouldn't hurt too much (cap the penalty)
    non_crit_count = len(missing) - crit_miss_count
    penalty_general = min(non_crit_count * 1, 10) # Max 10 pt penalty for fluff skills

    # --- FINAL CALC ---
    raw_score = (skill_score + exp_pts) - (penalty_critical + penalty_general)
    
    return max(0, min(100, raw_score)) 

def extract_skills(resume_text, max_tokens=512):
    # Function calling schema (manually enforced)
    schema_description = """
Your task is to extract ONLY the technical skills, experience and e-mail id from the user's resume.

Return JSON following exactly this schema:

Expected output:

{{
  "skills": ["Skill1", "Skill2", "Skill3"],
  "E-mail":"user's e-mail id",
  "experience": "Years of experience"
}}

Rules:
- ONLY return valid JSON.
- Remove the special characters.
- Do NOT add explanations or commentary.
- Expand abbreviations (CNN → Convolutional Neural Network).
- Include every technical skill found in the resume.
- Remove duplicates.
- If no skills found, return {{"skills": [], "E-mail":"user's e-mail id"}}
- If no email found return {{"skills": ["Skill1", "Skill2", "Skill3"], "E-mail":""}}
- If no skills and no email found return {{"skills": [], "E-mail":""}}
- If the user has no experience return the experience as 0.
"""

    prompt = schema_description + "\n\nResume text:\n" + resume_text

    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.1,        # deterministic
        do_sample=False
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #return response

    # Extract JSON from output
    try:
        json_start = response.rfind("{")
        json_end = response.rfind("}") + 1
        json_str = response[json_start:json_end]

        result = json.loads(json_str)
        return {"Status":"OK", "output":result}

    except Exception as e:
        return {"Status":"ERROR", "output":[str(e)]}
    
def summarize_jd(jd_text, max_tokens=512):
    prompt = f"""
You are a skilled job description generator and your job is to create a proper Job description for the following job description in the exact structure below.
DO NOT repeat the prompt or add any extra text.

Job Description:
{jd_text}

Output Structure:
# Job Title
- A one-line job name
# Job Summary
- A concise overview (3-4 lines)

# Key Responsibilities
- Bullet points

# Required Skills
- Bullet points

# Experience
- Years of experience required

# Preferred Skills
- Bullet points

# Skills (JSON)
{{"skills": ["skill1", "skill2", ...]}}

Rules:
- If no Experience mentioned, set it to 0
- If no skills mentioned, generate the technical skills based on the job description.
"""


    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.1,
        do_sample=False
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        _idx = summary.rfind("# Job Title")
        summary = summary[_idx+len("# Job Title"):]
        temp = summary[summary.rfind("# Experience")+len("# Experience"):summary.rfind('# Preferred Skills')-1]
        experience = int(temp.replace('\n', ''))

        skills_required = json.loads(summary[summary.rfind("# Skills")+len("# Skills"):])
        return {"Status":"OK", "outputs":[summary, skills_required, experience]}
    except Exception as e:
        return {"Status":"ERROR", "outputs":[str(e)]}

def chat(text, called_by):
    load_model()
    max_length = 512

    if called_by == "resume_skill_extractor":
        return extract_skills(text, max_length)
    
    elif called_by == "jd_generator":
        return summarize_jd(text, max_tokens=max_length) 
    else:
        return {"Status":"Error", "output": "Invalid called_by parameter"}


def skills_matcher(js, cs, cex, jex, max_tokens=512):

    prompt = f"""You are an expert ATS (Applicant Tracking System) scoring engine.
Your task is to compare candidate skills with job description skills and classify each skill according to how well they match.

You MUST return ONLY valid JSON using the exact format shown below.

Expected output:
{{
  "exact_matches": [],
  "semantic_matches": [],
  "partial_matches": [],
  "missing_skills": [],
}}

### Classification Rules
Classify each required job skill into one of the categories below:

1. EXACT MATCH:
   - Candidate skill matches the job skill word-for-word or very close variation.

2. SEMANTIC MATCH:
   - Meaning is the same even if wording differs.

3. PARTIAL MATCH:
   - Related but not fully covering the job requirement.

4. MISSING:
   - The job skill is not present in any form in the candidate skills.

### Input Data
Candidate Skills:
{cs}

Candidate Experience (years):
{cex}

Job Description Skills:
{js}

Required Experience (years):
{jex}
### Output data

"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.1,
        do_sample=False
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    

    # Extract JSON only
    json_start = response.rfind("### Output data")
    json_str = response[json_start+len("### Output data"):]

    try:
        return json.loads(json_str)
    except:
        return {"error": "Failed to parse JSON", "raw": response}
