import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle


# Load your trained model and the scaler
model = load_model("churn_model.h5")

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

#making use of the model to train input from the user
def customer_churn(customer_data):
    churn = model.predict(customer_data)
    return churn

st.title("Customer Churn Predictor")

st.write("Enter customer information: ")

customer_data = {}

#rewriting the names of the features clearly
features ={
    'Partner': 'Partner', 'Dependents':'Dependents','OnlineSecurity':'Online Security',
    'OnlineBackup':'Online Backup', 'DeviceProtection':'Device Protection','TechSupport':'Tech Support',
    'Contract':'Contract','PaperlessBilling':'Paperless Billing','SeniorCitizen':'SeniorCitizen',
    'tenure':'Tenure','MonthlyCharges':'Monthly Charges'

}


feature_questions = {
    'Partner': 'Do you have a partner?',
    'Dependents': 'Do you have dependents?',
    'OnlineSecurity': 'Do you use online security?',
    'OnlineBackup': 'Do you use online backup?',
    'DeviceProtection': 'Do you use device protection?',
    'TechSupport': 'Do you use tech support?',
    'Contract': 'What type of contract do you have?',
    'PaperlessBilling': 'Do you use paperless billing?',
    'SeniorCitizen': 'Are you a senior citizen?',
    'tenure': 'What is your tenure?',
    'MonthlyCharges': 'What are your monthly charges?'
}

numeric_features = ['SeniorCitizen','tenure', 'MonthlyCharges']
categorical_features = ['Partner', 'Dependents', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'Contract', 'PaperlessBilling']

#specifying the data types for the input data
for feature, question in feature_questions.items():
    if feature in ['tenure', 'MonthlyCharges']:
        customer_data[feature] = st.number_input(question, min_value=0.0, max_value=100.0)
    elif feature == 'SeniorCitizen':
     customer_data[feature] = st.selectbox(question, [0, 1])
    elif feature == 'Contract':
        contract_options = ["Month-to-month", "One year", "Two years"]
        customer_data[feature] = st.selectbox(question, contract_options)
    else:
        yes_no_options = ["Yes", "No"]
        customer_data[feature] = st.selectbox(question, yes_no_options)

#converting the inputs into a dataframe
customer_frame = pd.DataFrame([customer_data])

#scaling the numeric features 
customer_frame[numeric_features] = scaler.transform(customer_frame[numeric_features])

#encoding the categorical features
label_encoder = LabelEncoder()
for feature in categorical_features:
    customer_frame[feature] = label_encoder.fit_transform(customer_frame[feature])

#The accuracy score of the model
program_conf = 0.8380206663287787


#finally converting the dataframe into a numpy array then calling the prediction function and printing out the results
if st.button("Predict"):
    customer_data_array = customer_frame.values

    # Convert the input data to a DataFrame for prediction
    new_data = pd.DataFrame(customer_data_array, index=[0])

    # Call the prediction function
    churn = customer_churn(new_data)
    st.write(f"Predicted Customer Churn: {churn[0]}")
    st.write("Program Confidence: " + str(program_conf))




#     # Display the predicted player rating
#     st.write(f"Predicted Player Rating: {churn[0]}")
#     
