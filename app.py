import streamlit as st
import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -------------------------------
# Load & train model (v1)
# -------------------------------
df = pd.read_excel("chatgpt_ADHD_prediction.xlsx")

df = df.rename(columns={
    'ADHD Status': 'ADHD',
    'Grade/Percentage in exams/education': 'Grade',
    'Number  hours using mobile phone/tablet/smartphone per day(in hrs)': 'Mobile_Hours',
    'Number of hours spent watching TV in a day': 'TV_Hours',
    'BMI - Calculate by https://www.calculator.net/bmi-calculator.html': 'BMI',
    'No of hours of sleep in a day': 'Sleep_Hours'
})

df['ADHD'] = df['ADHD'].map({'ADHD': 1, 'Non-ADHD': 0})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Area'] = df['Area'].map({'Urban': 1, 'Rural': 0})
df['Family'] = df['Family'].map({'Nuclear': 1, 'Joint': 0})
df['Goes to college'] = df['Goes to college'].map({'Yes': 1, 'No': 0})

grade_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
df['Grade'] = df['Grade'].map(grade_map)

df = df.dropna()

X = df.drop(columns=['ADHD'])
y = df['ADHD']

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])

model.fit(X, y)

feature_names = X.columns
coefficients = model.named_steps['clf'].coef_[0]

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="ADHD Risk Estimation Tool")

st.title("ADHD Risk Estimation Tool")
st.markdown(
    "**For research and educational purposes only. Not a diagnostic tool.**"
)

st.header("Enter Details")

age = st.number_input("Age", 18, 30, 22)
gender = st.selectbox("Gender", ["Male", "Female"])
area = st.selectbox("Area", ["Urban", "Rural"])
family = st.selectbox("Family Type", ["Nuclear", "Joint"])
college = st.selectbox("Goes to College", ["Yes", "No"])
grade = st.selectbox("Academic Grade", ["A", "B", "C", "D"])
mobile = st.slider("Mobile phone usage (hours/day)", 0, 12, 4)
tv = st.slider("TV watching (hours/day)", 0, 10, 2)
sleep = st.slider("Sleep duration (hours/day)", 3, 12, 7)
bmi = st.number_input("BMI", 15.0, 40.0, 22.0)

if st.button("Estimate ADHD Risk"):

    input_data = np.array([[
        age,
        1 if gender == "Male" else 0,
        1 if family == "Nuclear" else 0,
        1 if area == "Urban" else 0,
        1 if college == "Yes" else 0,
        grade_map[grade],
        mobile,
        tv,
        bmi,
        sleep
    ]])

    probability = model.predict_proba(input_data)[0][1] * 100

    if probability < 30:
        risk = "Low"
    elif probability < 60:
        risk = "Moderate"
    else:
        risk = "High"

    st.subheader("Results")
    st.write(f"**Estimated likelihood of ADHD:** {probability:.1f}%")
    st.write(f"**Risk category:** {risk}")

    # Top contributing factors
    contributions = input_data[0] * coefficients
    top_idx = np.argsort(np.abs(contributions))[-3:][::-1]

    st.subheader("Top Contributing Factors")
    for i in top_idx:
        st.write(f"- {feature_names[i].replace('_', ' ')}")
