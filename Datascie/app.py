# app.py
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, mean_squared_error
import joblib

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('heart_attack.csv')

# Function to calculate the probability of having CVD
def calculate_probability(user_values, filtered_df, variables):
    probability = 0.0

    for var, user_val in zip(variables, user_values):
        if var == 'gender':
            threshold_val = filtered_df[var].mode().iloc[0]
            probability += int(str(user_val) == str(threshold_val))
        else:
            threshold_val = filtered_df[var].mean() + filtered_df[var].std()
            probability += int(float(user_val) > threshold_val) if user_val != '' else 0

    probability /= len(variables)
    return probability

# Function to train and evaluate models
def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear'),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True)
    }

    results = {}

    for name, model in models.items():
        if name == 'Logistic Regression':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            probabilities = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            probabilities = None

        joblib.dump(model, f'{name}_model.joblib')

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        results[name] = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mse': mse,
            'rmse': rmse,
            'probabilities': probabilities,
            'model': model
        }

    return results

# Display variable options and prompt the user for input
variables = ['age', 'gender', 'trestbps', 'cp', 'heart_disease']

# Define these variables at the beginning to make them accessible in the entire script
additional_variables = []
initial_probability = 0.0
model_results = {}

# Web App Routes
@app.route('/')
def index():
    return render_template('website.html', variables=variables)

@app.route('/predict_additional', methods=['POST'])
def predict_additional():
    global additional_variables
    global initial_probability
    global model_results

    user_values = []

    for var in variables:
        if var != 'gender':
            user_val = request.form.get(var, '')
            if user_val == '':
                user_values.append(np.nan)  # Set to NaN for numerical variables
            else:
                user_values.append(float(user_val))
        else:
            user_val = request.form.get(var, '')
            user_values.append(user_val)  # Append the original string, not the uppercase

    filtered_df = df[variables]

    # Train-test split for model evaluation
    X = df.drop('heart_disease', axis=1)
    y = df['heart_disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Evaluate models
    model_results = evaluate_models(X_train, X_test, y_train, y_test)

    # Calculate the probability of having CVD
    initial_probability = calculate_probability(user_values, filtered_df, variables)

    # Recommend additional information based on probability
    if initial_probability > 0.35:
        additional_variables = ['chol', 'fbs', 'restecg', 'thalach', 'thal']
    else:
        additional_variables = ['chol', 'fbs', 'restecg']

    return render_template('website.html', user_input={'initial_probability': initial_probability}, additional_variables=additional_variables, model_results=model_results)

@app.route('/predict_final', methods=['POST'])
def predict_final():
    global additional_variables
    global initial_probability
    global model_results

    user_values = []

    for var in variables + additional_variables:
        user_val = request.form.get(var, '')
        if user_val == '':
            user_values.append(np.nan)  # Set to NaN for numerical variables
        else:
            user_values.append(float(user_val) if var != 'gender' else user_val)  # For 'gender', keep it as a string

    filtered_df = df[variables + additional_variables]

    # Recalculate the probability with additional information
    updated_probability = calculate_probability(user_values, filtered_df, variables + additional_variables)

    # Check if probability increased by 10% or more
    if updated_probability - initial_probability >= 0.10:
        result_message = "HIGH CHANCE OF CVD"
    elif initial_probability - updated_probability >= 0.15:
        result_message = "LOW CHANCE OF CVD"
    else:
        result_message = "NO SPECIFIC RECOMMENDATIONS"

    # Create a dictionary to pass user input to the result.html
    user_input = {var: user_values[i] for i, var in enumerate(variables + additional_variables)}

    return render_template('result.html', result_message=result_message, user_input=user_input, model_results=model_results)

if __name__ == '__main__':
    app.run(debug=True)
