import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from datetime import datetime
from langsmith import traceable
import json
from groq import Groq
import os
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "employee-narrative-evaluator-system"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
@st.cache_resource
def download_nltk():
    nltk.download("punkt")

download_nltk()


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Narrative Evaluation System",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>

/* MAIN BACKGROUND */


/* SIDEBAR COLOR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#3b82c4,#2f6fa7);
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: white !important;
}

/* DROPDOWN STYLE */
div[data-baseweb="select"] {
    background-color: white;
    border-radius: 8px;
}

/* RESET BUTTON */
.stButton > button {
    background-color: #22c55e;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    width: 100%;
}

.stButton > button:hover {
    background-color: #16a34a;
}

/* INFO BOX */
.stAlert {
    background-color: #dbeafe;
    border-radius: 10px;
}
.block-container {
    padding-top: 2rem;
}

/* CENTER TITLE */
.main-title {
    text-align: center;
    font-size: 34px;
    font-weight: bold;
}

/* HR LINE */
hr {
    border: 1px solid #cbd5e1;
}
/* DOWNLOAD BUTTON */
div.stDownloadButton > button {
    background-color: #22c55e;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    border: none;
}

div.stDownloadButton > button:hover {
    background-color: #16a34a;
}

</style>
""", unsafe_allow_html=True)
col1, col2, col3 = st.columns([2,1,2])

with col2:
    st.image("images/llm.png", width=300)

st.markdown("""
<div style="text-align:center;">
<h1 style="font-size:40px;">Agentic AI Learning Program - Foundation Learning</h1>
<hr>
</div>""",unsafe_allow_html=True)



@traceable(name="employee_evaluation_llm")
def evaluate_with_llm(rule, evidence,status):
    
    prompt = f"""
You are evaluating an employee narrative.

Criterion:
{rule}

Evidence from narrative:
{evidence}

The evaluation result is: {status}

Explain briefly why the result is {status}.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}],
        temperature=0,
         
    )

    return response.choices[0].message.content
@traceable(name="suggestion_generation_llm")
def generate_suggestions(criteria, evidence):

    prompt = f"""
An employee failed the following performance criterion.

Criterion:
{criteria}

Evidence from narrative:
{evidence}

Provide 3 professional suggestions to improve performance.

Return short bullet points.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3,
        

    )

    return response.choices[0].message.content


def semantic_evaluation(rules, sentences, sentence_embeddings, embedding_model,narrative):

        results = []
        narrative_embedding = embedding_model.encode(narrative)
        used_indices = set()
        for rule, rule_text in rules.items():
        
            rule_embedding = embedding_model.encode(rule_text)
            similarities = cosine_similarity(
        [rule_embedding],
        sentence_embeddings
    )[0]
            best_index = similarities.argmax()

            evidence = sentences[best_index]
           
            # get top 3 sentences
            sorted_indices = similarities.argsort()[::-1]

            top_scores = similarities[sorted_indices[:3]]

            best_index = None

            for idx in sorted_indices:
                if idx not in used_indices:
                    best_index = idx
                    used_indices.add(idx)
                    break

            # fallback if all sentences already used
            if best_index is None:
                best_index = sorted_indices[0]

            evidence = sentences[best_index]

            best_score = similarities[best_index]

            score_10 = round(best_score * 10, 2)

            intent_score = f"{round(top_scores.mean() * 100, 2)}%"

            threshold = 5

            if score_10 >= threshold:
                status = "PASS"
            else:
                status = "FAIL"
            reason = evaluate_with_llm(rule_text, evidence, status)  

            results.append({
                "Criteria": rule,
                "Score (0-10)": score_10,
                "Intent Score(%)": intent_score,
                "Status": status,
                "Reason":reason,
                "Evidence": evidence
              
            })

        return pd.DataFrame(results)

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedding_model = load_model()

# ---------------- RUBRIC RULES ----------------

rules = {
"Innovation Impact": "Employee must demonstrate innovation with measurable impact.",
"Research Rigor": "Work should follow structured research methodology.",
"Strategic Alignment": "Work must align with organizational goals.",
"Measurable Outcomes": "Quantifiable results must be clearly mentioned.",
"Cross-Functional Collaboration": "Collaboration across teams must be demonstrated.",
"Problem Solving & Analytical Depth": "Complex problem solving must be shown.",
"Knowledge Dissemination": "Findings must be shared with stakeholders."
}

# ---------------- PDF READER ----------------

def read_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    return text

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown(
    "<h2 style='color:white;text-align: center; font-size:30px;'>Solution Scope</h2>",
    unsafe_allow_html=True)
    application = st.selectbox(
        "Select Application",
        ["Select Application", "Employee Narrative Evaluation"]
    )

    model = st.selectbox(
        "LLM Models",
        ["Select Model", "Llama 3.3 70B"]
    )

    framework = st.selectbox(
        "LLM Framework",
        ["Select Framework", "Groq API"]
    )

    gcp = st.selectbox(
        "GCP Services Used",
        ["Select Service", "Cloud Run", "Secret Manager"]
    )
    st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)
    if st.button("Clear/Reset",use_container_width=True):
        st.rerun()
    st.markdown(
    "<div style='text-align: center;color:white;'>Build & Deployed on</div>",
    unsafe_allow_html=True
)
    col1, col2, col3,col4 = st.columns(4)

    with col1:
        st.image("images/llm.png", width=60)

    with col2:
        st.image("images/google.png", width=60)

    with col3:
        st.image("images/aws.png", width=60)
    with col4:
        st.image("images/azure.png",width=60)



# ---------------- MAIN APPLICATION ----------------

if application == "Employee Narrative Evaluation":

    st.header("Employee Narrative Evaluation System")

    uploaded_file = st.file_uploader(
        "Upload Employee Narrative PDF",
        type=["pdf"]
    )

    # ---------------- RUN EVALUATION ----------------

    if uploaded_file and st.button("Run Evaluation"):

        narrative = read_pdf(uploaded_file)

        sentences = sent_tokenize(narrative)

        sentence_embeddings = embedding_model.encode(sentences)

        df = semantic_evaluation(
            rules,
            sentences,
            sentence_embeddings,
            embedding_model,
            narrative
        )

        st.session_state["evaluation"] = df


    # ---------------- DISPLAY RESULTS ----------------

    if "evaluation" in st.session_state:

        df = st.session_state["evaluation"]

        st.subheader("Semantic Evaluation Results")

        st.dataframe(df, use_container_width=True)

        # ---------------- HUMAN REVIEW ----------------

        st.subheader("Human Review (Human-in-the-Loop)")

        human_results = []

        for i, row in df.iterrows():

            st.markdown(f"### {row['Criteria']}")

            decision = st.selectbox(
                "Human Decision",
                ["PASS", "FAIL"],
                index=0 if row["Status"] == "PASS" else 1,
                key=f"decision_{i}"
            )

            feedback = ""

            # Show feedback box ONLY if decision changed
            if decision != row["Status"]:

                feedback = st.text_area(
                    "Feedback",
                    placeholder="Explain the reason for changing the AI decision...",
                    key=f"feedback_{i}"
                )

            human_results.append({
                "Criteria": row["Criteria"],
                "Score": row["Score (0-10)"],
                "AI Status": row["Status"],
                "Human Decision": decision,
                "Human Feedback": feedback,
                "Evidence": row["Evidence"]
            })

        human_df = pd.DataFrame(human_results)

        # ---------------- FINALIZE EVALUATION ----------------

        if st.button("Finalize Evaluation"):

            pass_count = (human_df["Human Decision"] == "PASS").sum()

            total = len(human_df)

            percentage = round((pass_count / total) * 100, 2)

            st.subheader("Final Performance Score")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=percentage,
                number={'suffix': "%"},
                gauge={'axis': {'range': [0, 100]}}
            ))

            st.plotly_chart(fig, use_container_width=True)

            if percentage >= 70:
                st.success("MEETS STANDARD")
            else:
                st.error("DOES NOT MEET STANDARD")

        # ---------------- IMPROVEMENT SUGGESTIONS ----------------

            failed_criteria = human_df[human_df["Human Decision"] == "FAIL"]

            if len(failed_criteria) > 0:

                st.subheader("Improvement Recommendations")

                failed_rows = human_df[human_df["Human Decision"] == "FAIL"]

                for i, row in failed_rows.iterrows():

                    st.markdown(f"### {row['Criteria']}")

                    suggestions = generate_suggestions(
                        row["Criteria"],
                        row["Evidence"]
                    )

                    st.write(suggestions)

        # ---------------- DOWNLOAD REPORT ----------------

                csv = human_df.to_csv(index=False).encode("utf-8")

                filename = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

                st.download_button(
                        "Download Report",
                        data=csv,
                        file_name=filename,
                        mime="text/csv"
                    )




