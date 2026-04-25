import streamlit as st
import pandas as pd
import speech_recognition as sr
import PyPDF2
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Career Chat Pro", page_icon="🤖", layout="wide")

# -------------------------------
# UI (ChatGPT STYLE)
# -------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0b0f19;
    color: white;
}

.user {
    background: #2563eb;
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
    max-width: 75%;
    margin-left: auto;
}

.ai {
    background: #1f2937;
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
    max-width: 75%;
}

.card {
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    padding: 15px;
    border-radius: 12px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Career Chat Pro (Voice + Resume + ChatGPT Style)")
st.markdown("Talk, upload resume, or type — AI will guide your career")

# -------------------------------
# LOAD DATA + TRAIN MODEL
# -------------------------------
df = pd.read_csv("career_data.csv")

le_i = LabelEncoder()
le_s = LabelEncoder()
le_p = LabelEncoder()
le_c = LabelEncoder()

df['Interest_enc'] = le_i.fit_transform(df['Interest'])
df['Skill_enc'] = le_s.fit_transform(df['Skill'])
df['Personality_enc'] = le_p.fit_transform(df['Personality'])
df['Career_enc'] = le_c.fit_transform(df['Career'])

X = df[['Interest_enc','Skill_enc','Personality_enc']]
y = df['Career_enc']

model = RandomForestClassifier()
model.fit(X, y)

accuracy = cross_val_score(model, X, y, cv=5).mean()

# -------------------------------
# SESSION STATE
# -------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("🎤 Voice + Resume")

    # ---------------- VOICE INPUT ----------------
    if st.button("🎤 Speak (Voice Input)"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening... speak now 🎧")
            audio = r.listen(source)

        try:
            text = r.recognize_google(audio)
            st.success(f"You said: {text}")
            st.session_state.voice_input = text
        except:
            st.error("Could not understand audio")

    # ---------------- RESUME UPLOAD ----------------
    uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

    resume_text = ""
    if uploaded_file:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            resume_text += page.extract_text()

        st.success("Resume uploaded successfully!")

        st.text_area("Extracted Resume Text", resume_text, height=200)

# -------------------------------
# DISPLAY CHAT
# -------------------------------
for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.markdown(f"<div class='user'>👤 {msg['text']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='ai'>🤖 {msg['text']}</div>", unsafe_allow_html=True)

# -------------------------------
# CHAT INPUT
# -------------------------------
voice_text = st.session_state.get("voice_input", "")

user_input = st.chat_input("Type or use voice...") or voice_text

# -------------------------------
# FEATURE EXTRACTION (SMART AI)
# -------------------------------
def extract_features(text):
    text = text.lower()

    interest = df['Interest'].iloc[0]
    skill = df['Skill'].iloc[0]
    personality = df['Personality'].iloc[0]

    for i in df['Interest'].unique():
        if i.lower() in text:
            interest = i

    for s in df['Skill'].unique():
        if s.lower() in text:
            skill = s

    for p in df['Personality'].unique():
        if p.lower() in text:
            personality = p

    return interest, skill, personality

# -------------------------------
# RESUME SKILL EXTRACTION (SIMPLE AI LOGIC)
# -------------------------------
def resume_to_profile(text):
    text = text.lower()

    interest = "AI"
    skill = "Python"
    personality = "Analytical"

    if "data" in text:
        skill = "SQL"
        interest = "Data Science"

    if "developer" in text:
        skill = "Python"
        interest = "Software Development"

    if "team" in text:
        personality = "Extrovert"
    else:
        personality = "Analytical"

    return interest, skill, personality

# -------------------------------
# RESPONSE ENGINE
# -------------------------------
if user_input:

    st.session_state.chat.append({"role": "user", "text": user_input})

    # merge chat + resume
    if resume_text:
        r_interest, r_skill, r_personality = resume_to_profile(resume_text)
    else:
        r_interest, r_skill, r_personality = extract_features(user_input)

    input_data = [[
        le_i.transform([r_interest])[0],
        le_s.transform([r_skill])[0],
        le_p.transform([r_personality])[0]
    ]]

    prediction = model.predict(input_data)
    probs = model.predict_proba(input_data)[0]

    career = le_c.inverse_transform(prediction)[0]
    confidence = max(probs) * 100

    response = f"""
🎯 Career Match: {career}

📊 Confidence: {confidence:.2f}%

🧠 Based on:
- Interest: {r_interest}
- Skill: {r_skill}
- Personality: {r_personality}

📄 Resume: {'Used' if resume_text else 'Not uploaded'}

📈 Model Accuracy: {accuracy*100:.2f}%
"""

    # typing effect
    placeholder = st.empty()
    temp = ""

    for c in response:
        temp += c
        placeholder.markdown(f"<div class='ai'>🤖 {temp}</div>", unsafe_allow_html=True)
        time.sleep(0.01)

    st.session_state.chat.append({"role": "ai", "text": response})

    st.session_state.voice_input = ""
    
