from flask import Flask, render_template, request, redirect
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('heart_attack.csv')

# Function to preprocess user input
def preprocess_input(age, gender, trestbps, cp, heart_disease):
    gender = 0 if gender.upper() == 'M' else 1
    return pd.DataFrame([[age, gender, trestbps, cp, heart_disease]],
                        columns=['age', 'gender', 'trestbps', 'cp', 'heart_disease'])

# Function to scale input features
def scale_input(X, scaler):
    return scaler.transform(X)

# Function to train and evaluate models
def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
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
            probabilities = None  # KNN does not have a predict_proba method

        results[name] = {
            'prediction': y_pred.tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'model': model
        }

    return results

# Add route for additional information
@app.route('/additional_info', methods=['POST'])
def additional_info():
    # Process the form data for additional information
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalac = float(request.form['thalac'])
    thal = float(request.form['thal'])

    # Further processing or display as needed

    return "Additional information processed successfully!"

# Modify the existing predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Process the form data
    age = float(request.form['age'])
    gender = request.form['gender']
    trestbps = float(request.form['trestbps'])
    cp = float(request.form['cp'])
    heart_disease = float(request.form['heart_disease'])  # Change here
    user_input = preprocess_input(age, gender, trestbps, cp, heart_disease)

    # Debugging statements
    print(f"Input values: {user_input}")
    
    # Train-test split for model evaluation
    X = df.drop('heart_disease', axis=1)
    y = df['heart_disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Evaluate models
    model_results = evaluate_models(X_train, X_test, y_train, y_test)

    # Debugging statements
    print(f"Probabilities: {model_results['Logistic Regression']['probabilities']}")

    # Get the initial probability
    initial_probability = model_results['Logistic Regression']['probabilities'][0]

    # Debugging statements
    print(f"Initial Probability: {initial_probability}")

    # Determine which result template to render based on probability
    if initial_probability > 0.35:
        user_input['probability'] = initial_probability * 100  # Convert to percentage
        return render_template('result_high_cvd.html', model_results=model_results, user_input=user_input)
    else:
        user_input['probability'] = initial_probability * 100  # Convert to percentage
        return render_template('result_low_cvd.html', model_results=model_results, user_input=user_input)

# Default route for the initial form
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Process the form data
        age = float(request.form['age'])
        gender = request.form['gender']
        trestbps = float(request.form['trestbps'])
        cp = float(request.form['cp'])
        has_history = float(request.form['has_history'])
        user_input = preprocess_input(age, gender, trestbps, cp, has_history)

        # Train-test split for model evaluation
        X = df.drop('heart_disease', axis=1)
        y = df['heart_disease']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Evaluate models
        model_results = evaluate_models(X_train, X_test, y_train, y_test)

        # Get the initial probability
        initial_probability = model_results['Logistic Regression']['probabilities'][0]

        # Determine which result template to render based on probability
        if initial_probability > 0.35:
            return render_template('result_high_cvd.html', model_results=model_results, user_input=user_input)
        else:
            return render_template('result_low_cvd.html', model_results=model_results, user_input=user_input)

    return render_template('website.html')

if __name__ == '__main__':
    app.run(debug=True)
    