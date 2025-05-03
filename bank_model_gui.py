import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# --------- 1. Load & Train Model ---------
# Load dataset with correct separator and quote handling
data = pd.read_csv(r"C:\Users\Disha Kadam\OneDrive\Desktop\bank.csv", sep=';', quotechar='"')

# Encode target variable
data_encoded = pd.get_dummies(data.drop('y', axis=1), drop_first=True)
data_encoded['y'] = data['y'].map({'yes': 1, 'no': 0})

# Split into X and y
X = data_encoded.drop('y', axis=1)
y = data_encoded['y']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Save the model
joblib.dump(clf, 'bank_marketing_model.joblib')

# --------- 2. GUI ---------
# Load model
model = joblib.load('bank_marketing_model.joblib')

# Define mappings for input encoding
job_options = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
               'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed']
marital_options = ['divorced', 'married', 'single']
education_options = ['primary', 'secondary', 'tertiary']
binary_options = ['no', 'yes']
contact_options = ['cellular', 'telephone', 'unknown']

def encode_inputs(age, job, marital, education, default, housing, loan, contact, duration):
    # Create a dict from inputs and match order used in training
    input_dict = {
        'age': int(age),
        'duration': int(duration),
        'job_' + job: 1,
        'marital_' + marital: 1,
        'education_' + education: 1,
        'default_yes': 1 if default == 'yes' else 0,
        'housing_yes': 1 if housing == 'yes' else 0,
        'loan_yes': 1 if loan == 'yes' else 0,
        'contact_' + contact: 1
    }

    # Initialize input vector
    input_vector = []
    for col in X.columns:
        input_vector.append(input_dict.get(col, 0))

    return input_vector

# Tkinter App
app = tk.Tk()
app.title("Bank Marketing Subscription Predictor")
app.geometry("500x700")
app.configure(bg="#f0f0f0")

# --- UI Labels and Inputs ---
def add_label_input(label_text, options=None):
    label = tk.Label(app, text=label_text, font=('Arial', 12), bg="#f0f0f0")
    label.pack(pady=2)
    if options:
        cb = ttk.Combobox(app, values=options, state='readonly')
        cb.pack(pady=5)
        return cb
    else:
        entry = tk.Entry(app)
        entry.pack(pady=5)
        return entry

title = tk.Label(app, text="🧠 Bank Marketing Subscription Predictor", font=("Arial", 16, 'bold'), bg="#f0f0f0")
title.pack(pady=15)

age_input = add_label_input("Age:")
job_input = add_label_input("Job:", job_options)
marital_input = add_label_input("Marital Status:", marital_options)
education_input = add_label_input("Education Level:", education_options)
default_input = add_label_input("Default Credit?", binary_options)
housing_input = add_label_input("Housing Loan?", binary_options)
loan_input = add_label_input("Personal Loan?", binary_options)
contact_input = add_label_input("Contact Type:", contact_options)
duration_input = add_label_input("Last Contact Duration (in seconds):")

# --- Prediction Result ---
result_label = tk.Label(app, text="", font=('Arial', 14, 'bold'), bg="#f0f0f0")
result_label.pack(pady=20)

# --- Predict Function ---
def predict():
    try:
        inputs = encode_inputs(
            age_input.get(),
            job_input.get(),
            marital_input.get(),
            education_input.get(),
            default_input.get(),
            housing_input.get(),
            loan_input.get(),
            contact_input.get(),
            duration_input.get()
        )
        pred = model.predict([inputs])[0]
        conf = model.predict_proba([inputs])[0][pred]

        if pred == 1:
            result_label.config(text=f"✔️ Likely to Subscribe (Confidence: {conf:.2f})", fg="green")
        else:
            result_label.config(text=f"❌ Not Likely to Subscribe (Confidence: {conf:.2f})", fg="red")
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}", fg="red")

# --- Button ---
predict_btn = tk.Button(app, text="Predict", font=('Arial', 12), bg="#007acc", fg="white", command=predict)
predict_btn.pack(pady=10)

# Run App
app.mainloop()
