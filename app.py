import streamlit as st
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Career Guide", page_icon="🤖", layout="centered")

# ---------------- TYPING ANIMATION ----------------
def type_writer(text, speed=0.02):
    placeholder = st.empty()
    typed = ""

    for char in text:
        typed += char
        placeholder.markdown(f'<div class="bot-box">{typed}</div>', unsafe_allow_html=True)
        time.sleep(speed)

# ---------------- CSS ----------------
st.markdown("""
<style>

.block-container {
    max-width: 600px;
    padding-top: 2rem;
}

.title {
    font-size: 26px;
    font-weight: 700;
    text-align: center;
}

.user-box {
    background: #2563eb;
    color: white;
    padding: 10px;
    border-radius: 12px;
    margin: 6px 0;
    text-align: right;
}

.bot-box {
    background: rgba(128,128,128,0.15);
    padding: 10px;
    border-radius: 12px;
    margin: 6px 0;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="title">🤖 AI Career Guide</div>', unsafe_allow_html=True)
st.caption("Find your best career match instantly")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("career_data.csv")

# ---------------- LABEL ENCODING ----------------
le_i = LabelEncoder()
le_s = LabelEncoder()
le_p = LabelEncoder()
le_c = LabelEncoder()

df['Interest_enc'] = le_i.fit_transform(df['Interest'])
df['Skill_enc'] = le_s.fit_transform(df['Skill'])
df['Personality_enc'] = le_p.fit_transform(df['Personality'])
df['Career_enc'] = le_c.fit_transform(df['Career'])

X = df[['Interest_enc', 'Skill_enc', 'Personality_enc']]
y = df['Career_enc']

# ---------------- MODEL ----------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model.fit(X, y)

# ---------------- SMART RULE SYSTEM (IMPORTANT FIX) ----------------
def rule_based_career(skill):
    skill = skill.lower()

    if "power bi" in skill:
        return "Power BI Analyst"
    elif "excel" in skill:
        return "MIS Analyst"
    elif "python" in skill:
        return "Data Scientist"
    elif "sql" in skill:
        return "Data Analyst"
    else:
        return None  # fallback to ML

# ---------------- SESSION STATE ----------------
if "step" not in st.session_state:
    st.session_state.step = 1

# ---------------- STEP 1 ----------------
if st.session_state.step == 1:
    st.markdown('<div class="bot-box">What are you interested in?</div>', unsafe_allow_html=True)

    interest = st.selectbox("", df['Interest'].unique())

    if st.button("Next"):
        st.session_state.interest = interest
        st.session_state.step = 2

# ---------------- STEP 2 ----------------
elif st.session_state.step == 2:
    st.markdown(f'<div class="user-box">{st.session_state.interest}</div>', unsafe_allow_html=True)
    st.markdown('<div class="bot-box">What is your skill?</div>', unsafe_allow_html=True)

    skill = st.selectbox("", df['Skill'].unique())

    if st.button("Next"):
        st.session_state.skill = skill
        st.session_state.step = 3

# ---------------- STEP 3 ----------------
elif st.session_state.step == 3:
    st.markdown(f'<div class="user-box">{st.session_state.interest}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="user-box">{st.session_state.skill}</div>', unsafe_allow_html=True)
    st.markdown('<div class="bot-box">What is your personality?</div>', unsafe_allow_html=True)

    personality = st.selectbox("", df['Personality'].unique())

    if st.button("Get Result"):
        st.session_state.personality = personality
        st.session_state.step = 4

# ---------------- RESULT ----------------
elif st.session_state.step == 4:

    st.markdown(f'<div class="user-box">{st.session_state.interest}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="user-box">{st.session_state.skill}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="user-box">{st.session_state.personality}</div>', unsafe_allow_html=True)

    skill = st.session_state.skill.lower()
    interest = st.session_state.interest.lower()

    # ---------------- SMART CAREER ENGINE ----------------
    if "power bi" in skill:
        career = "Power BI Analyst"

    elif "excel" in skill:
        career = "MIS Analyst / Reporting Analyst"

    elif "python" in skill:
        career = "Data Scientist / ML Engineer"

    elif "sql" in skill:
        career = "Data Analyst"

    elif "finance" in interest:
        career = "Financial Analyst"

    elif "marketing" in interest:
        career = "Digital Marketing Analyst"

    else:
        career = "Business Analyst"

    # ---------------- OUTPUT ----------------
    with st.spinner("Analyzing your profile..."):
        time.sleep(1)

    type_writer(f"🎯 Your best career is: {career}")
    type_writer("Based on your skills + interest + personality match.", 0.02)

    if st.button("🔄 Start Again"):
        st.session_state.step = 1
    # ---------------- RULE FIRST (FIX) ----------------
    career = rule_based_career(st.session_state.skill)

    # ---------------- IF RULE FAILS → ML ----------------
    if career is None:

        input_data = [[
            le_i.transform([st.session_state.interest])[0],
            le_s.transform([st.session_state.skill])[0],
            le_p.transform([st.session_state.personality])[0]
        ]]

        pred = model.predict(input_data)
        career = le_c.inverse_transform(pred)[0]

    # ---------------- THINKING ----------------
    with st.spinner("Analyzing your profile..."):
        time.sleep(1)

    # ---------------- OUTPUT ----------------
    type_writer(f"🎯 Your best career is: {career}")

    type_writer("This is based on your interest, skill and personality match.", 0.015)

    if st.button("🔄 Start Again"):
        st.session_state.step = 1

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("✨ Built by Vaishnavi Nadar")
