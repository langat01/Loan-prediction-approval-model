from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/loan_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        gender = int(request.form['gender'])
        married = int(request.form['married'])
        dependents = int(request.form['dependents'])
        education = int(request.form['education'])
        self_employed = int(request.form['self_employed'])
        applicant_income = float(request.form['applicant_income'])
        coapplicant_income = float(request.form['coapplicant_income'])
        loan_amount = float(request.form['loan_amount'])
        loan_term = float(request.form['loan_term'])
        credit_history = int(request.form['credit_history'])
        property_area = int(request.form['property_area'])

        # Create feature array
        features = np.array([[
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_term, credit_history, property_area
        ]])

        # Make prediction
        prediction = model.predict(features)

        # Return result
        result = "Approved" if prediction[0] == 1 else "Not Approved"
        return render_template('result.html', prediction=result)
    
    except Exception as e:
        return str(e)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        features = np.array([
            data['gender'],
            data['married'],
            data['dependents'],
            data['education'],
            data['self_employed'],
            data['applicant_income'],
            data['coapplicant_income'],
            data['loan_amount'],
            data['loan_term'],
            data['credit_history'],
            data['property_area']
        ]).reshape(1, -1)
        
        prediction = model.predict(features)
        result = 'Loan Approved' if prediction[0] == 1 else 'Loan Rejected'
        
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
