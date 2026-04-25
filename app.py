import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Career Recommender", page_icon="🚀", layout="centered")

# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
    <div style='text-align:center'>
        <h1>🚀 AI Career Recommendation System</h1>
        <p style='color:gray;'>Discover your perfect career using AI</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("career_data.csv")

# -------------------------------
# ENCODING
# -------------------------------
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

# -------------------------------
# MODEL (Random Forest)
# -------------------------------
model = RandomForestClassifier()
model.fit(X, y)

accuracy = cross_val_score(model, X, y, cv=5).mean()

# -------------------------------
# INPUT UI
# -------------------------------
st.markdown("### 🧠 Your Profile")

col1, col2, col3 = st.columns(3)

with col1:
    interest = st.selectbox("🎯 Interest", df['Interest'].unique())

with col2:
    skill = st.selectbox("💻 Skill", df['Skill'].unique())

with col3:
    personality = st.selectbox("🧩 Personality", df['Personality'].unique())

# -------------------------------
# BUTTON
# -------------------------------
st.markdown("<br>", unsafe_allow_html=True)

center = st.columns([1,2,1])
with center[1]:
    clicked = st.button("🚀 Analyze My Career")

# -------------------------------
# RESULT
# -------------------------------
if clicked:
    with st.spinner("🤖 AI is analyzing your profile..."):

        input_data = [[
            le_i.transform([interest])[0],
            le_s.transform([skill])[0],
            le_p.transform([personality])[0]
        ]]

        prediction = model.predict(input_data)
        probs = model.predict_proba(input_data)[0]

        career_ml = le_c.inverse_transform(prediction)[0]
        confidence = max(probs) * 100

        # -------------------------------
        # RESULT CARD
        # -------------------------------
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #6366F1, #8B5CF6);
            padding:25px;
            border-radius:15px;
            color:white;
            text-align:center;
        ">
            <h2>🎯 {career_ml}</h2>
            <p>Best Career Match for You</p>
        </div>
        """, unsafe_allow_html=True)

        # -------------------------------
        # CONFIDENCE
        # -------------------------------
        st.write("### 📊 Confidence Score")
        st.progress(int(confidence))
        st.write(f"{confidence:.2f}% match")

        # -------------------------------
        # PLOTLY CHART (TOP 3)
        # -------------------------------
        st.write("### 📊 Career Match Visualization")

        top_idx = probs.argsort()[-3:][::-1]
        careers = [le_c.inverse_transform([i])[0] for i in top_idx]
        values = [probs[i]*100 for i in top_idx]

        chart_df = pd.DataFrame({
            "Career": careers,
            "Match %": values
        })

        fig = px.bar(chart_df, x="Career", y="Match %", title="Top Career Matches")
        st.plotly_chart(fig)

        # -------------------------------
        # MODEL PERFORMANCE
        # -------------------------------
        st.write("### 📈 Model Accuracy")
        st.write(f"{accuracy*100:.2f}%")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("""
    <div style='text-align:center; color:gray;'>
        ✨ Built by Vaishnavi Nadar • AI Career Recommender
    </div>
""", unsafe_allow_html=True)
