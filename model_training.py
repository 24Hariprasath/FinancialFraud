# model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib  # To save the trained model

# Load and preprocess the data
data = pd.read_csv("/Users/shree/Downloads/sample.csv")

data["type"] = data["type"].map({
    "CASH_OUT": 1,
    "PAYMENT": 2,
    "CASH_IN": 3,
    "TRANSFER": 4,
    "DEBIT": 5
})

data["isFraud"] = data["isFraud"].map({
    0: "No Fraud",
    1: "Fraud"
})

x = data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]]
y = data["isFraud"]

# Split data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)

# Save the trained model
joblib.dump(model, 'saved_model.pkl')
print("Model saved as saved_model.pkl")
