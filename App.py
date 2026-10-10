import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson

st.set_page_config(page_title="AI Football Correct Score Predictor", layout="wide")

st.title("📊 AI Football Correct Score Predictor")
st.write("ระบบวิเคราะห์และคำนวณเปอร์เซ็นต์สกอร์ฟุตบอลด้วยหลักการ Poisson Distribution")

# Sidebar สำหรับปรับตั้งค่า
st.sidebar.header("⚙️ ตั้งค่าข้อมูลการแข่งขัน")
home_team = st.sidebar.text_input("ชื่อทีมเหย้า (Home Team)", "Home Team")
away_team = st.sidebar.text_input("ชื่อทีมเยือน (Away Team)", "Away Team")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 ค่าสถิติประตูที่คาดหวัง (Expected Goals - xG)")
home_xG = st.sidebar.slider(f"xG ของ {home_team}", 0.0, 5.0, 1.8, 0.1)
away_xG = st.sidebar.slider(f"xG ของ {away_team}", 0.0, 5.0, 1.2, 0.1)

max_goals = 5
score_matrix = np.zeros((max_goals + 1, max_goals + 1))

for home_goals in range(max_goals + 1):
    for away_goals in range(max_goals + 1):
        home_prob = poisson.pmf(home_goals, home_xG)
        away_prob = poisson.pmf(away_goals, away_xG)
        score_matrix[home_goals, away_goals] = home_prob * away_prob

df_scores = pd.DataFrame(
    score_matrix * 100, 
    columns=[f"{g}" for g in range(max_goals + 1)],
    index=[f"{g}" for g in range(max_goals + 1)]
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔮 5 อันดับสกอร์ที่มีโอกาสเกิดมากที่สุด")
    all_scores = df_scores.unstack()
    top_5 = all_scores.sort_values(ascending=False).head(5)
    
    for index, prob in top_5.items():
        away, home = index
        st.info(f"⚽ สกอร์ **{home} - {away}** : มีโอกาสเกิดขึ้น **{prob:.2f}%**")

with col2:
    st.subheader("📈 วิเคราะห์อัตรา สูง/ต่ำ & BTTS")
    home_win_p = np.tril(score_matrix, -1).sum() * 100
    draw_p = np.diag(score_matrix).sum() * 100
    away_win_p = np.triu(score_matrix, 1).sum() * 100
    
    st.write(f"**โอกาสชนะ (1X2 Outlook):**")
    st.write(f"🏠 {home_team} ชนะ: {home_win_p:.2f}% | 🤝 เสมอ: {draw_p:.2f}% | 🚀 {away_team} ชนะ: {away_win_p:.2f}%")
    
    # Over/Under 2.5
    under_25 = score_matrix[0,0] + score_matrix[0,1] + score_matrix[0,2] + score_matrix[1,0] + score_matrix[1,1] + score_matrix[2,0]
    over_25 = 1.0 - under_25
    
    st.markdown("---")
    st.write(f"**อัตราประตูรวม สูง/ต่ำ (Over/Under 2.5):**")
    st.write(f"🟢 สูงกว่า 2.5 ประตู: {over_25*100:.2f}% | 🔴 ต่ำกว่า 2.5 ประตู: {under_25*100:.2f}%")

st.markdown("---")
st.subheader("🧮 ตารางเปอร์เซ็นต์ความน่าจะเป็นแบบละเอียด (Probability Matrix Heatmap)")
st.dataframe(df_scores.style.background_gradient(cmap="Greens").format("{:.2f}%"))
st.caption("แกนนอน: สกอร์ทีมเยือน | แกนตั้ง: สกอร์ทีมเหย้า (หน่วยเป็นเปอร์เซ็นต์)")
