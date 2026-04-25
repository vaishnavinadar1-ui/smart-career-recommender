import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Career Recommender",
    page_icon="🚀",
    layout="centered"
)

# -------------------------------
# HEADER (RESPONSIVE)
# -------------------------------
st.markdown("""
<div style="text-align:center; padding:10px;">
    <h1 style="font-size:32px;">🚀 AI Career Recommendation System</h1>
    <p style="color:gray; font-size:14px;">Discover your perfect career using AI</p>
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

X = df[['Interest_enc', 'Skill_enc', 'Personality_enc']]
y = df['Career_enc']

# -------------------------------
# MODEL
# -------------------------------
model = RandomForestClassifier()
model.fit(X, y)

accuracy = cross_val_score(model, X, y, cv=5).mean()

# -------------------------------
# RESPONSIVE INPUT UI
# -------------------------------
st.markdown("### 🧠 Your Profile")

is_mobile = st.session_state.get("mobile", False)

# Detect mobile by screen width trick (simple fallback)
# Streamlit can't truly detect screen size, so we keep safe layout

col1 = st.container()
col2 = st.container()
col3 = st.container()

interest = st.selectbox("🎯 Interest", df['Interest'].unique())
skill = st.selectbox("💻 Skill", df['Skill'].unique())
personality = st.selectbox("🧩 Personality", df['Personality'].unique())

# -------------------------------
# BUTTON
# -------------------------------
clicked = st.button("🚀 Analyze My Career")

# -------------------------------
# RESULT
# -------------------------------
if clicked:
    with st.spinner("AI is analyzing your profile..."):

        input_data = [[
            le_i.transform([interest])[0],
            le_s.transform([skill])[0],
            le_p.transform([personality])[0]
        ]]

        prediction = model.predict(input_data)
        probs = model.predict_proba(input_data)[0]

        career_ml = le_c.inverse_transform(prediction)[0]
        confidence = max(probs) * 100

        # RESULT CARD
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #6366F1, #8B5CF6);
            padding:20px;
            border-radius:15px;
            color:white;
            text-align:center;
        ">
            <h2>{career_ml}</h2>
            <p>Best Career Match</p>
        </div>
        """, unsafe_allow_html=True)

        # CONFIDENCE
        st.write("### 📊 Confidence Score")
        st.progress(int(confidence))
        st.write(f"{confidence:.2f}% match")

        # TOP CAREERS
        st.write("### 📊 Top Matches")

        top_idx = probs.argsort()[-3:][::-1]
        careers = [le_c.inverse_transform([i])[0] for i in top_idx]
        values = [probs[i] * 100 for i in top_idx]

        fig_df = pd.DataFrame({
            "Career": careers,
            "Match %": values
        })

        fig = px.bar(fig_df, x="Career", y="Match %")
        st.plotly_chart(fig, use_container_width=True)

        # ACCURACY
        st.write("### 📈 Model Accuracy")
        st.write(f"{accuracy * 100:.2f}%")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("<center style='color:gray;'>Built by Vaishnavi Nadar</center>", unsafe_allow_html=True)
