import streamlit as st
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Career AI Chat", page_icon="🤖", layout="wide")

# -------------------------------
# CHATGPT STYLE UI
# -------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0b0f19;
    color: white;
}

/* Chat bubbles */
.user {
    background: #2563eb;
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
    max-width: 75%;
    margin-left: auto;
    color: white;
}

.ai {
    background: #1f2937;
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
    max-width: 75%;
    color: white;
}

/* Title */
.title {
    text-align:center;
    font-size:28px;
    font-weight:700;
}

.subtitle {
    text-align:center;
    color:gray;
    margin-bottom:15px;
}

/* Card */
.card {
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    padding: 15px;
    border-radius: 12px;
    text-align:center;
    color:white;
    margin-top:10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<div class='title'>🤖 AI Career Coach</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Talk like ChatGPT & discover your ideal career</div>", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR (CHATGPT STYLE SETTINGS)
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🧹 Clear Chat"):
        st.session_state.chat = []
        st.rerun()

    st.markdown("---")
    st.write("📊 Model: Random Forest")
    st.write("🧠 Type: Career Predictor AI")
    st.write("🚀 Version: Pro Chat UI")

# -------------------------------
# LOAD DATA
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
# SESSION MEMORY
# -------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# -------------------------------
# DISPLAY CHAT HISTORY
# -------------------------------
for msg in st.session_state.chat:
    if msg["role"] == "user":
        st.markdown(f"<div class='user'>👤 {msg['text']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='ai'>🤖 {msg['text']}</div>", unsafe_allow_html=True)

# -------------------------------
# CHAT INPUT
# -------------------------------
user_input = st.chat_input("Ask me about your career... (e.g. I like coding, AI, introvert)")

# -------------------------------
# SMART PARSER
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
# RESPONSE ENGINE (WITH TYPING EFFECT)
# -------------------------------
if user_input:

    st.session_state.chat.append({"role": "user", "text": user_input})

    interest, skill, personality = extract_features(user_input)

    input_data = [[
        le_i.transform([interest])[0],
        le_s.transform([skill])[0],
        le_p.transform([personality])[0]
    ]]

    prediction = model.predict(input_data)
    probs = model.predict_proba(input_data)[0]

    career = le_c.inverse_transform(prediction)[0]
    confidence = max(probs) * 100

    response_text = f"""
🎯 Best Career Match: {career}

📊 Confidence: {confidence:.2f}%

🧠 Your Profile:
- Interest: {interest}
- Skill: {skill}
- Personality: {personality}

📈 Model Accuracy: {accuracy*100:.2f}%
"""

    # -------------------------------
    # TYPING EFFECT (ChatGPT FEEL)
    # -------------------------------
    placeholder = st.empty()
    typed_text = ""

    for char in response_text:
        typed_text += char
        placeholder.markdown(f"<div class='ai'>🤖 {typed_text}</div>", unsafe_allow_html=True)
        time.sleep(0.01)

    st.session_state.chat.append({"role": "ai", "text": response_text})
