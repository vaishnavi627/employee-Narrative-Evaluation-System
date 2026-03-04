import streamlit as st
from PyPDF2 import PdfReader
from groq import Groq
from dotenv import load_dotenv
import os
import pandas as pd
import json
from datetime import datetime
import plotly.graph_objects as go
import os
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Executive Employee Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🏢 Executive Evaluation")
    st.markdown("---")
    st.markdown("### 📌 System Overview")
    st.write("AI-powered structured employee evaluation system.")
    st.markdown("---")
    st.markdown("### 📋 Evaluation Criteria")
    for rule in rules:
        st.markdown(f"✔️ {rule}")
    st.markdown("---")
    if st.button("🔄 Reset"):
        st.rerun()

# ---------------- PROFESSIONAL CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #111827);
    color: #e5e7eb;
    font-family: 'Segoe UI', sans-serif;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 25px;
    color: #38bdf8;
}
.card {
    background-color: #1e293b;
    padding: 28px;
    border-radius: 14px;
    margin-top: 25px;
    border: 1px solid #334155;
    box-shadow: 0 4px 15px rgba(0,0,0,0.5);
}
.dataframe {
    background-color: #0f172a;
    border-radius: 10px;
}
.dataframe thead tr {
    background-color: #0ea5e9;
    color: white;
}
.badge-pass {
    background-color: #16a34a;
    padding: 6px 14px;
    border-radius: 20px;
    color: white;
    font-weight: 600;
}
.badge-fail {
    background-color: #dc2626;
    padding: 6px 14px;
    border-radius: 20px;
    color: white;
    font-weight: 600;
}
.final-pass {
    text-align: center;
    font-size: 24px;
    font-weight: 700;
    color: #22c55e;
}
.final-fail {
    text-align: center;
    font-size: 24px;
    font-weight: 700;
    color: #ef4444;
}
.stFileUploader {
    background-color: #0f172a !important;
    border: 2px solid #0ea5e9 !important;
    border-radius: 12px;
    padding: 20px;
}
.stButton > button {
    background-color: #0ea5e9;
    color: white;
    font-weight: 600;
    border-radius: 8px;
}
.stButton > button:hover {
    background-color: #0284c7;
}
/* DOWNLOAD BUTTON - MATCH PRIMARY BUTTON COLOR */
.stDownloadButton > button {
    background-color: #0ea5e9 !important;
    color: white !important;
    font-weight: 600;
    border-radius: 8px;
}

.stDownloadButton > button:hover {
    background-color: #0284c7 !important;
}
section[data-testid="stSidebar"] {
    background-color: #0f172a !important;
}
section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}
/* PREMIUM SUGGESTION BOX */
.suggestion-box {
    background: linear-gradient(145deg, #0f172a, #1e293b);
    border-left: 5px solid #22d3ee;
    padding: 25px;
    border-radius: 12px;
    margin-top: 15px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    line-height: 1.7;
    font-size: 16px;
}

.suggestion-box h1,
.suggestion-box h2,
.suggestion-box h3 {
    color: #38bdf8;
}

.suggestion-box strong {
    color: #22d3ee;
}

.suggestion-box ul {
    margin-left: 20px;
}

.suggestion-box li {
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD API ----------------

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.markdown('<div class="title">Employee Narrative Evaluation System</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload Employee Narrative PDF", type=["pdf"])

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

if uploaded_file:
    narrative = read_pdf(uploaded_file)

    if st.button("🚀 Run Evaluation"):
        with st.spinner("AI is evaluating performance..."):

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
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📊 Decision Summary")
            st.markdown(df.to_html(index=False), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ---------------- REASONING SUMMARY ----------------
            reasoning_rows = []
            for key, value in data.items():
                status = "PASS" if value["result"] == "YES" else "FAIL"
                badge = '<span class="badge-pass">PASS</span>' if status=="PASS" else '<span class="badge-fail">FAIL</span>'
                reasoning_rows.append({
                    "Qualification": key,
                    "What the Rule Requires": rules.get(key, ""),
                    "Narrative Evidence": value["evidence"],
                    "Status": badge
                })

            reasoning_df = pd.DataFrame(reasoning_rows)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📋 Detailed Reasoning Summary")
            st.markdown(reasoning_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ---------------- SCORE ----------------
            yes_count = (df["Result"] == "YES").sum()
            total = len(df)
            percentage = round((yes_count / total) * 100, 2)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📈 Performance Overview")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=percentage,
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#0ea5e9"},
                    'bgcolor': "#0f172a"
                }
            ))

            fig.update_layout(
                paper_bgcolor="#1e293b",
                plot_bgcolor="#1e293b",
                font={'color': "#e5e7eb"},
                height=350
            )

            st.plotly_chart(fig, use_container_width=True)

            if yes_count == total:
                st.markdown('<div class="final-pass">✅ MEETS STANDARD</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="final-fail">❌ DOES NOT MEET STANDARD</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # ---------------- SUGGESTIONS SECTION ----------------
                        # ---------------- PREMIUM SUGGESTIONS SECTION ----------------
            failed_criteria = [k for k, v in data.items() if v["result"] == "NO"]

            if failed_criteria:
                suggestion_prompt = f"""
            Provide professional, structured, well-formatted improvement recommendations 
            for the following failed criteria:
            {failed_criteria}

            Use headings and bullet points.
            """

                suggestion_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": suggestion_prompt}],
                    temperature=0.3
                )

                suggestions_text = suggestion_response.choices[0].message.content

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("💡 Improvement Recommendations")

                st.markdown(
                    f'<div class="suggestion-box">{suggestions_text}</div>',
                    unsafe_allow_html=True
                )

                st.markdown('</div>', unsafe_allow_html=True)
            # ---------------- EXPORT ----------------
            # ---------------- SAVE TO EVALUATION HISTORY ----------------

            history_file = "evaluation_history.csv"

            history_data = df.copy()
            history_data["Score"] = percentage
            history_data["Passed"] = yes_count
            history_data["Total Criteria"] = total
            history_data["Evaluation Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if os.path.exists(history_file):
                history_data.to_csv(history_file, mode="a", header=False, index=False)
            else:
                history_data.to_csv(history_file, index=False)

            # ---------------- DOWNLOAD CSV ----------------

            csv = df.to_csv(index=False).encode("utf-8")

            filename = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            st.download_button(
                "📥 Download Report",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
            # ---------------- SHOW HISTORY ----------------

            if os.path.exists("evaluation_history.csv"):

                history_df = pd.read_csv("evaluation_history.csv", on_bad_lines="skip")
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📊 Evaluation History")

                st.markdown(history_df.to_html(index=False), unsafe_allow_html=True)


                st.markdown('</div>', unsafe_allow_html=True)