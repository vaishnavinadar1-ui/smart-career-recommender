import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Career Recommender",
    page_icon="🚀",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("🚀 AI Career Recommendation System")
st.caption("Discover your perfect career using AI")

st.markdown("---")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("career_data.csv")

# ---------------- USER INPUT ----------------
st.subheader("🧠 Your Profile")

interest = st.selectbox("🎯 Select your Interest", df['Interest'].unique())
skill = st.selectbox("💻 Select your Skill", df['Skill'].unique())
personality = st.selectbox("🧩 Select your Personality", df['Personality'].unique())

st.markdown("---")

# ---------------- ML MODEL ----------------
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

    st.markdown("### 🤖 AI Result")

    # Encode user input
    input_data = [[
        le_interest.transform([interest])[0],
        le_skill.transform([skill])[0],
        le_personality.transform([personality])[0]
    ]]

    # Prediction
    prediction = model.predict(input_data)
    career_ml = le_career.inverse_transform(prediction)[0]

    # Probabilities
    probs = model.predict_proba(input_data)[0]

    # ---------------- OUTPUT ----------------
    st.success(f"🎯 Recommended Career: **{career_ml}**")

    confidence = max(probs) * 100
    st.write(f"📊 Confidence Score: **{confidence:.2f}%**")

    # ---------------- CHART ----------------
    st.write("### 📈 Career Probability Distribution")

    prob_df = pd.DataFrame({
        "Career": le_career.classes_,
        "Probability": probs
    }).sort_values(by="Probability", ascending=False)

    fig = px.bar(prob_df, x="Career", y="Probability")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- INSIGHTS ----------------
    st.write("### 🧠 AI Insights")

    if interest == "Data":
        st.write("👉 You are inclined towards data-driven roles")

    if skill in ["Python", "SQL", "Excel"]:
        st.write("👉 You have strong analytical/technical skills")

    if personality == "Analytical":
        st.write("👉 Analytical mindset suits problem-solving careers")

    # ---------------- CAREER INFO ----------------
    career_info = {
        "Data Analyst": "Works with data to generate insights and support business decisions.",
        "Business Analyst": "Bridges business needs with data-driven solutions.",
        "ML Engineer": "Builds machine learning models and AI systems.",
        "Marketing Manager": "Handles brand promotion and marketing strategies."
    }

    st.write("### 📘 About this Career")
    st.info(career_info.get(career_ml, "No description available"))

    # ---------------- SUMMARY ----------------
    st.write("### 🎯 Why this fits you")

    st.write(f"""
    ✔ Based on your interest in **{interest}**  
    ✔ Your skill in **{skill}**  
    ✔ Your **{personality}** personality  
    👉 This career aligns well with your profile
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("✨ Built by Vaishnavi Nadar | AI Career Recommender")
