import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("logistic_regression.joblib")

# Title of the app
st.title('Income Prediction :ship:')

# Define the input fields
st.header("Enter Input Features")

Age = st.number_input("Age", min_value=0, max_value=100)
Workclass = st.selectbox("Work Class", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
Final_Weight = st.number_input("Final Weight", min_value=0, max_value=200000)
EducationNum = st.number_input("Education Number", min_value=0, max_value=20)
Marital_Status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
Occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
Relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
Race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
Gender = st.selectbox("Gender", ['Male', 'Female'])
Capital_Gain = st.number_input("Capital Gain", min_value=0)
Capital_Loss = st.number_input("Capital Loss", min_value=0)
Hours_per_Week = st.number_input("Hours per Week", min_value=0, max_value=100)

# List of input columns
columns = ['Age', 'Workclass', 'Final Weight', 'EducationNum', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Gender', 'Capital Gain', 'capital loss', 'Hours per Week']

# Define predict function
def predict():
    # Convert inputs into a DataFrame
    row = np.array([Age, Workclass, Final_Weight, EducationNum, Marital_Status, Occupation, Relationship, Race, Gender, Capital_Gain, Capital_Loss, Hours_per_Week])
    X = pd.DataFrame([row], columns=columns)

    # Ensure correct data types
    X = X.convert_dtypes()

    # Make prediction
    prediction = model.predict(X)

    # Display prediction result
    if prediction[0] == 1:
        st.success('Predicted Income: >50K :thumbsup:')
    else:
        st.error('Predicted Income: <=50K :thumbsdown:')

# Trigger prediction
trigger = st.button('Predict', on_click=predict)


