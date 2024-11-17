# app.py
from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib
import psycopg2

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('saved_model.pkl')

# Connect to PostgreSQL database
conn = psycopg2.connect(
    host="localhost",
    database="Transactions",
    user="postgres",
    password="1111"
)
cursor = conn.cursor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    type_val = int(data['type'])
    amount = float(data['amount'])
    oldbalanceOrg = float(data['oldbalance'])
    newbalanceOrg = float(data['newbalance'])

    # Make prediction
    input_data = np.array([[type_val, amount, oldbalanceOrg, newbalanceOrg]])
    prediction = model.predict(input_data)[0]

    if prediction == "No Fraud":
        prediction = 0
    elif prediction == "Fraud":
        prediction = 1

    try:
        cursor.execute(
            "INSERT INTO transaction_history (type, amount, oldbalanceOrg, newbalanceOrg, isFraud) VALUES (%s, %s, %s, %s, %s)",
            (type_val, amount, oldbalanceOrg, newbalanceOrg, prediction)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()  # Rollback the transaction if any error occurs
        print(f"Error inserting data: {e}")

    # Send prediction result as JSON
    return jsonify({'isFraud': prediction})

if __name__ == '__main__':
    app.run(debug=True)
