from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load and preprocess data (only needed once)
def load_data():
    df = pd.read_csv('data/train.csv')
    
    # Fill missing values
    categorical_cols = df.select_dtypes(include='object').columns.drop('Loan_ID')
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    return df

# Load model
model = joblib.load('model/loan_prediction_model.pkl')

# Homepage
@app.route('/')
def home():
    return render_template('index.html')

# Prediction form handler
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = {
            'Gender': int(request.form['gender']),
            'Married': int(request.form['married']),
            'Dependents': int(request.form['dependents']),
            'Education': int(request.form['education']),
            'Self_Employed': int(request.form['self_employed']),
            'ApplicantIncome': float(request.form['applicant_income']),
            'CoapplicantIncome': float(request.form['coapplicant_income']),
            'LoanAmount': float(request.form['loan_amount']),
            'Loan_Amount_Term': float(request.form['loan_term']),
            'Credit_History': int(request.form['credit_history']),
            'Property_Area': int(request.form['property_area'])
        }
        
        # Convert to DataFrame for prediction
        input_df = pd.DataFrame([form_data])
        prediction = model.predict(input_df)[0]
        
        # Prepare result
        result = "Approved" if prediction == 1 else "Not Approved"
        return render_template('result.html', 
                             prediction=result,
                             form_data=form_data)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

# API endpoint
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        required_fields = [
            'Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
            'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
            'Property_Area'
        ]
        
        # Validate input
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Prepare features
        features = pd.DataFrame([data])
        prediction = model.predict(features)[0]
        
        return jsonify({
            'prediction': 'Approved' if prediction == 1 else 'Not Approved',
            'probability': float(model.predict_proba(features)[0][1])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
