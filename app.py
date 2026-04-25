import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Career App", page_icon="🚀", layout="centered")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Background */
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

/* Main container */
.block-container {
    padding-top: 2rem;
}

/* Title */
.title {
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: white;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 14px;
    margin-bottom: 20px;
}

/* Card style */
.card {
    background: #1e293b;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}

/* Button */
.stButton>button {
    background-color: #6366f1;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 16px;
}

/* Success box */
.stSuccess {
    border-radius: 10px;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    .title {
        font-size: 22px;
    }
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="title">🚀 AI Career Recommendation System</div>
<div class="subtitle">Discover your perfect career using AI</div>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("career_data.csv")

# ---------------- INPUT CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("🧠 Your Profile")

interest = st.selectbox("🎯 Interest", df['Interest'].unique())
skill = st.selectbox("💻 Skill", df['Skill'].unique())
personality = st.selectbox("🧩 Personality", df['Personality'].unique())

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MODEL ----------------
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

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# ---------------- BUTTON ----------------
if st.button("🔍 Get Recommendation"):

    input_data = [[
        le_interest.transform([interest])[0],
        le_skill.transform([skill])[0],
        le_personality.transform([personality])[0]
    ]]

    prediction = model.predict(input_data)
    career_ml = le_career.inverse_transform(prediction)[0]
    probs = model.predict_proba(input_data)[0]

    # ---------------- RESULT CARD ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.success(f"🎯 Recommended Career: {career_ml}")
    st.write(f"📊 Confidence: {max(probs)*100:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- CHART CARD ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.write("📈 Career Probability")

    prob_df = pd.DataFrame({
        "Career": le_career.classes_,
        "Probability": probs
    }).sort_values(by="Probability", ascending=False)

    fig = px.bar(prob_df, x="Career", y="Probability")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- INSIGHTS CARD ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.write("🧠 AI Insights")

    if interest == "Data":
        st.write("👉 You prefer data-driven roles")

    if skill in ["Python", "SQL", "Excel"]:
        st.write("👉 Strong analytical/technical skills detected")

    if personality == "Analytical":
        st.write("👉 Analytical mindset fits problem-solving careers")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div style="text-align:center; color:gray; margin-top:20px;">
✨ Built by Vaishnavi Nadar
</div>
""", unsafe_allow_html=True)
