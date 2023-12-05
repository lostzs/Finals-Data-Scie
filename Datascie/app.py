from flask import Flask, render_template, request
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Process the form data
        age = float(request.form['age'])
        gender = request.form['gender']
        trestbps = float(request.form['trestbps'])
        cp = float(request.form['cp'])
        heart_disease = None  # Update this with the actual form input for heart_disease
        user_input = preprocess_input(age, gender, trestbps, cp, heart_disease)

        # Train-test split for model evaluation
        X = df.drop('heart_disease', axis=1)
        y = df['heart_disease']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Evaluate models
        model_results = evaluate_models(X_train, X_test, y_train, y_test)

        # Your existing logic for further processing and displaying results

        return render_template('result.html', model_results=model_results, user_input=user_input)

    return render_template('website.html')

if __name__ == '__main__':
    app.run(debug=True)
