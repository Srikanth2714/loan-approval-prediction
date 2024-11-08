import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Load the dataset directly
df = pd.read_csv('loan_prediction.csv')
df.drop('Loan_ID',axis=1,inplace=True)

# Fill missing values in the dataset
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# Prepare data for modeling
le = LabelEncoder()
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
for i in cat_cols:
    df[i] = le.fit_transform(df[i])

# Define features and target variable
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Train the model using SVC (Support Vector Classifier)
model = SVC(random_state=42)
model.fit(X, y)

# Set up the Streamlit app layout
st.title("Loan Prediction App")

# User input fields for loan prediction
st.subheader("Enter Loan Application Details")

gender = st.selectbox("Gender", options=["Male", "Female"])
married = st.selectbox("Marital Status", options=["Yes", "No"])
dependents = st.selectbox("Dependents", options=["0", "1", "2", "3+"])
education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", options=["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.selectbox("Loan Amount Term (in months)", options=[360, 240, 180, 120, 60])
credit_history = st.selectbox("Credit History", options=[1.0, 0.0])
property_area = st.selectbox("Property Area", options=['Semiurban','Urban','Rural'])

# Convert user inputs to a DataFrame for prediction
input_data = {
    'Gender': 1 if gender == "Male" else 0,
    'Married': 1 if married == "Yes" else 0,
    'Dependents': int(dependents) if dependents.isdigit() else 3,  # Assuming "3+" maps to 3
    'Education': 1 if education == "Not Graduate" else 0,
    'Self_Employed': 1 if self_employed == "Yes" else 0,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_amount_term,
    'Credit_History': credit_history,
    'Property_Area': le.transform([property_area])[0]  # Encode Property_Area using LabelEncoder
}
# Ensure input data contains all columns from X
input_df = pd.DataFrame([input_data])
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing columns with default value 0


# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    loan_status = "Approved" if prediction == 1 else "Not Approved"
    st.subheader(f"Loan Status: {loan_status}")
