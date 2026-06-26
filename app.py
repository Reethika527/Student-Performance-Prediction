import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("student_performance_updated_1000.csv")

# Remove missing values
df = df.dropna()

# Title
st.title("Advanced Student Performance Prediction")

# Show dataset
st.write(df.head())

# Features
X = df[['StudyHoursPerWeek',
        'AttendanceRate',
        'PreviousGrade',
        'ExtracurricularActivities',
        'Attendance (%)']]

# Target
y = df['FinalGrade']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor()

model.fit(X_train, y_train)

# Accuracy
score = r2_score(y_test, model.predict(X_test))

st.write(f"Model Accuracy: {score:.2f}")

# User Inputs
study_hours = st.number_input(
    "Study Hours Per Week",
    min_value=0.0,
    key="study"
)

attendance = st.number_input(
    "Attendance Rate",
    min_value=0.0,
    key="attendance"
)

previous_grade = st.number_input(
    "Previous Grade",
    min_value=0.0,
    key="previous"
)

activities = st.number_input(
    "Extracurricular Activities",
    min_value=0.0,
    key="activities"
)

attendance_percent = st.number_input(
    "Attendance Percentage",
    min_value=0.0,
    key="percent"
)

# Prediction
if st.button("Predict Final Grade"):

    input_data = [[
        study_hours,
        attendance,
        previous_grade,
        activities,
        attendance_percent
    ]]

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Final Grade: {prediction[0]:.2f}"
    )