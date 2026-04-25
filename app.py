import streamlit as st
import pandas as pd

st.title("🚀 Career Recommendation System")

try:
    st.write("Step 1: Loading dataset...")

    df = pd.read_csv("career_data.csv")
    st.write("✅ Dataset loaded")

    st.write("Step 2: Creating inputs...")

    interest = st.selectbox("Interest", df['Interest'].unique())
    skill = st.selectbox("Skill", df['Skill'].unique())
    personality = st.selectbox("Personality", df['Personality'].unique())

    if st.button("Get Recommendation"):

        st.write("Step 3: Calculating score...")

        df['Score'] = 0

        for i in range(len(df)):
            score = 0

            if df.loc[i, 'Interest'] == interest:
                score += 2
            if df.loc[i, 'Skill'] == skill:
                score += 2
            if df.loc[i, 'Personality'] == personality:
                score += 1

            df.loc[i, 'Score'] = score

        top = df.sort_values(by='Score', ascending=False).iloc[0]

        st.success(f"Recommended Career: {top['Career']}")
        st.write("Score:", top['Score'])

except Exception as e:
    st.error("❌ ERROR FOUND:")
    st.write(e)
