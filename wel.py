# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the file path
file_path = r'C:\Users\Admin\Downloads\train.csv'

# Load the dataset
df = pd.read_csv(file_path)

# View the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Structure of the dataset
print("\nDataset Information:")
df.info()

# Basic statistical description
print("\nStatistical Summary:")
print(df.describe())

# Check columns with missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())
#categorical columns
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                   'Self_Employed', 'Property_Area', 'Loan_Status']

# Check value counts for each
for col in categorical_cols:
    print(f"\n{col}:\n{df[col].value_counts(dropna=False)}")
#numerical columns
num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
           'Loan_Amount_Term', 'Credit_History']

print(df[num_cols].describe())
# Fill categorical columns with mode
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill numerical columns with median
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())
# We need convert text into number for machine to understand 
from sklearn.preprocessing import LabelEncoder

# Initialize label encoder
le = LabelEncoder()

# Encode all categorical columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("\nData types after encoding:")
print(df.dtypes)
#data spliting into featuring and target
# Split into X (features) and y (target)
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']
from sklearn.model_selection import train_test_split

# Split the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)
# model evaluation
from sklearn.metrics import accuracy_score, classification_report

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# confusion matrix to understand how well the model is distinguishing between the classes
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrixSS
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
cm_display.plot(cmap='Blues')
# Example new data (you can change these values!)
new_application = {
    'Gender': 0,           # 1 = Male, 0 = Female (after encoding)
    'Married': 1,          # 1 = Married, 0 = Not Married
    'Dependents': 0,
    'Education': 0,        # 0 = Graduate, 1 = Not Graduate
    'Self_Employed': 0,
    'ApplicantIncome': 1000,
    'CoapplicantIncome': 2000,
    'LoanAmount': 150,
    'Loan_Amount_Term': 360,
    'Credit_History': 1,
    'Property_Area': 2     # 2 = Urban, 1 = Semiurban, 0 = Rural
}

# Convert to DataFrame
new_app_df = pd.DataFrame([new_application])

# Predict
new_prediction = model.predict(new_app_df)

# Decode prediction (1 = Approved, 0 = Not Approved)
if new_prediction[0] == 1:
    print("✅ Loan Approved!")
else:
    print("❌ Loan Not Approved!")
import matplotlib.pyplot as plt
import numpy as np

# Get feature importance
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
#saving our model
import joblib

# Save model to file
joblib.dump(model, 'loan_prediction_model.pkl')

print("Model saved successfully!")
import pickle

# Save the trained model
pickle.dump(model, open('loan_model.pkl', 'wb'))
print('saved')
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('loan_prediction_model.pkl')

# Create a Flask app
app = Flask(__name__)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)

        # Extract features
        features = np.array(data['features']).reshape(1, -1)

        # Predict
        prediction = model.predict(features)

        # Map prediction to label (optional)
        result = 'Loan Approved' if prediction[0] == 1 else 'Loan Rejected'

        # Return result as JSON
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
