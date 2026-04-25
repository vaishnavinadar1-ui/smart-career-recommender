import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Career Guide", page_icon="🤖")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    max-width: 500px;
}

/* Title */
.title {
    font-size: 24px;
    font-weight: 600;
    text-align: center;
}

/* Chat bubbles */
.user-box {
    background: #2563eb;
    color: white;
    padding: 10px;
    border-radius: 12px;
    margin: 5px 0;
    text-align: right;
}

.bot-box {
    background: #f1f5f9;
    padding: 10px;
    border-radius: 12px;
    margin: 5px 0;
}

/* Button */
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="title">🤖 AI Career Guide</div>', unsafe_allow_html=True)
st.caption("Find your best career match in seconds")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("career_data.csv")

# ---------------- ML ----------------
le_interest = LabelEncoder()
le_skill = LabelEncoder()
le_personality = LabelEncoder()
le_career = LabelEncoder()

df['Interest_enc'] = le_interest.fit_transform(df['Interest'])
df['Skill_enc'] = le_skill.fit_transform(df['Skill'])
df['Personality_enc'] = le_personality.fit_transform(df['Personality'])
df['Career_enc'] = le_career.fit_transform(df['Career'])

X = df[['Interest_enc', 'Skill_enc', 'Personality_enc']]
y = df['Career_enc']

model = RandomForestClassifier()
model.fit(X, y)

# ---------------- SESSION ----------------
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

    # Prediction
    input_data = [[
        le_interest.transform([st.session_state.interest])[0],
        le_skill.transform([st.session_state.skill])[0],
        le_personality.transform([st.session_state.personality])[0]
    ]]

    pred = model.predict(input_data)
    career = le_career.inverse_transform(pred)[0]

    st.markdown(f'<div class="bot-box">🎯 Your best career is <b>{career}</b></div>', unsafe_allow_html=True)

    if st.button("Start Again"):
        st.session_state.step = 1
