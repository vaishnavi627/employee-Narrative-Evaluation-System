import streamlit as st
from PyPDF2 import PdfReader
from groq import Groq
import pandas as pd
import json
from datetime import datetime
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Agentic AI Learning Program",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
col1, col2, col3 = st.columns([2,1,2])

with col2:
    st.image("images/llm.png", width=300)
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
# ---------------- EVALUATION RULES ----------------
rules = {
    "Innovation Impact": "Employee must demonstrate innovation with measurable impact.",
    "Research Rigor": "Work should follow structured research methodology.",
    "Strategic Alignment": "Work must align with organizational goals.",
    "Measurable Outcomes": "Quantifiable results must be clearly mentioned.",
    "Cross-Functional Collaboration": "Collaboration across teams must be demonstrated.",
    "Problem Solving & Analytical Depth": "Complex problem solving must be shown.",
    "Knowledge Dissemination": "Findings must be shared with stakeholders."
}
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

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3,col4 = st.columns(4)

    with col1:
        st.image("images/llm.png", width=60)

    with col2:
        st.image("images/google.png", width=60)

    with col3:
        st.image("images/aws.png", width=60)
    with col4:
        st.image("images/azure.png",width=60)
col1, col2, col3 = st.columns(3)


st.markdown("""
<div style="text-align:center;">
<h1 style="font-size:30px;">Autonomous Document Intelligence,Powered by Agentic AI</h1>
<hr>
</div>""",unsafe_allow_html=True)

if application == "Select Application":
    col1,col2 = st.columns([3,1])
    with col1:
        st.header("Select Application")
    with col2:
        st.info("Please select the application from the sidebar to continue")

# ---------------- MAIN APPLICATION ----------------
if application == "Employee Narrative Evaluation":

    st.header("Employee Narrative Evaluation System")

    uploaded_file = st.file_uploader(
        "Upload Employee Narrative PDF",
        type=["pdf"]
    )

    load_dotenv()

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def read_pdf(file):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    if uploaded_file:

        narrative = read_pdf(uploaded_file)

        if st.button("Run Evaluation"):

            with st.spinner("AI evaluating performance..."):

                prompt = f"""
Evaluate the employee narrative strictly using this rubric.

Return STRICT JSON format:

{{
"Innovation Impact": {{"result": "YES/NO", "evidence": ""}},
"Research Rigor": {{"result": "YES/NO", "evidence": ""}},
"Strategic Alignment": {{"result": "YES/NO", "evidence": ""}},
"Measurable Outcomes": {{"result": "YES/NO", "evidence": ""}},
"Cross-Functional Collaboration": {{"result": "YES/NO", "evidence": ""}},
"Problem Solving & Analytical Depth": {{"result": "YES/NO", "evidence": ""}},
"Knowledge Dissemination": {{"result": "YES/NO", "evidence": ""}}
}}

Narrative:
{narrative}
"""

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    response_format={"type": "json_object"}
                )

                data = json.loads(response.choices[0].message.content)

                rows = []
                for key, value in data.items():
                    rows.append({
                        "Criteria": key,
                        "Result": value["result"],
                        "Evidence": value["evidence"]
                    })

                df = pd.DataFrame(rows)

                # ---------------- DECISION SUMMARY ----------------
                st.subheader("Decision Summary")
                st.dataframe(df, use_container_width=True)

                # ---------------- REASONING ----------------
                reasoning_rows = []

                for key, value in data.items():

                    status = "PASS" if value["result"] == "YES" else "FAIL"

                    reasoning_rows.append({
                        "Qualification": key,
                        "Rule Requirement": rules.get(key, ""),
                        "Narrative Evidence": value["evidence"],
                        "Status": status
                    })

                reasoning_df = pd.DataFrame(reasoning_rows)

                st.subheader("Detailed Reasoning")
                st.dataframe(reasoning_df, use_container_width=True)

                # ---------------- SCORE ----------------
                yes_count = (df["Result"] == "YES").sum()
                total = len(df)
                percentage = round((yes_count / total) * 100, 2)

                st.subheader("Performance Score")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=percentage,
                    number={'suffix': "%"},
                    gauge={
                        'axis': {'range': [0, 100]}
                    }
                ))

                fig.update_layout(height=350)

                st.plotly_chart(fig, use_container_width=True)

                if yes_count == total:
                    st.success("MEETS STANDARD")
                else:
                    st.error("DOES NOT MEET STANDARD")

                # ---------------- SUGGESTIONS ----------------
                failed = [k for k, v in data.items() if v["result"] == "NO"]

                if failed:

                   suggestion_prompt = f"""
Provide professional improvement recommendations for EACH failed criterion separately.

Failed Criteria:
{failed}

Instructions:
- For every criterion, create a separate section.
- Show the criterion name as a heading.
- Under each heading, provide 2–3 bullet point recommendations.
- Each bullet point must be on a new line.
- Keep recommendations short and professional.

Example format:

Innovation Impact:
- recommendation
- recommendation

Research Rigor:
- recommendation
- recommendation
"""

                suggestion_response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": suggestion_prompt}]
                    )

                suggestions = suggestion_response.choices[0].message.content

                st.subheader("Improvement Recommendations")
                st.write(suggestions)
                csv = df.to_csv(index=False).encode("utf-8")
                
                filename = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.download_button(
                    "Download Report",
                    data=csv,
                    file_name=filename,
                    mime="text/csv",
                    
                )

                # # ---------------- SAVE HISTORY ----------------
                # 


                
                # history_file = "evaluation_history.csv"

                # history_data = df.copy()
                # history_data["Score"] = percentage
                # history_data["Evaluation Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # if os.path.exists(history_file):
                #     history_data.to_csv(history_file, mode="a", header=False, index=False)
                # else:
                #     history_data.to_csv(history_file, index=False)

                # # ---------------- DOWNLOAD ----------------
               

                # # ---------------- HISTORY TABLE ----------------
                # if os.path.exists(history_file):

                #     st.subheader("Evaluation History")

                #     history_df = pd.read_csv(history_file)

                #     st.dataframe(history_df, use_container_width=True)
