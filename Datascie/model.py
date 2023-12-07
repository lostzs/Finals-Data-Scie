import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, mean_squared_error
import seaborn as sns

# Load the dataset
df = pd.read_csv('heart_attack.csv')

# Function to calculate the probability of having CVD
def calculate_probability(user_values, filtered_df, variables):
    # Simple probability calculation (just for demonstration)
    # You should use an appropriate probability model based on your dataset and domain knowledge
    # This is a simple example assuming a threshold for high probability

    probability = 0.0

    for var, user_val in zip(variables, user_values):
        if var == 'gender':
            # Handle gender separately
            threshold_val = filtered_df[var].mode().iloc[0]  # Use mode for categorical variable
            probability += int(user_val == threshold_val)
        else:
            threshold_val = filtered_df[var].mean() + filtered_df[var].std()
            probability += int(user_val > threshold_val)

    probability /= len(variables)  # Normalize probability

    return probability

# Function to train and evaluate models
def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear'),  # Use 'liblinear' solver
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True)
    }

    results = {}

    for name, model in models.items():
        if name == 'Logistic Regression':
            # Use scaled data for Logistic Regression
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            probabilities = model.predict_proba(X_test_scaled)[:, 1]
        else:
            # Use original data for other models
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            probabilities = None  # KNN does not have a predict_proba method

        # Calculate evaluation metrics
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
user_values = []

# Prompt the user for input values
for var in variables:
    if var == 'gender':
        user_val = input(f"Enter gender (M/F): ").upper()
    else:
        user_val = float(input(f"Enter value for {var}: "))
    user_values.append(user_val)

# Filter the dataset based on selected variables
filtered_df = df[variables]

# Train-test split for model evaluation
X = df.drop('heart_disease', axis=1)
y = df['heart_disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate models
model_results = evaluate_models(X_train, X_test, y_train, y_test)

# Calculate the probability of having CVD
initial_probability = calculate_probability(user_values, filtered_df, variables)

print(f"Initial Probability of having CVD: {initial_probability:.2%}")

# Recommend additional information based on probability
if initial_probability > 0.35:
    additional_variables = ['chol', 'fbs', 'restecg', 'thalach', 'thal']
    print("Recommend getting additional information:")
else:
    additional_variables = ['chol', 'fbs', 'restecg']
    print("Recommend getting the following information:")

for var in additional_variables:
    if var == 'gender':
        user_val = input(f"Enter gender (M/F): ").upper()
    else:
        user_val = float(input(f"Enter value for {var}: "))
    user_values.append(user_val)

# Update the filtered dataset with additional information
filtered_df = df[variables + additional_variables]

# Recalculate the probability with additional information
updated_probability = calculate_probability(user_values, filtered_df, variables + additional_variables)
print(f"Updated Probability of having CVD: {updated_probability:.2%}")

# Check if probability increased by 10% or more
if updated_probability - initial_probability >= 0.10:
    print("Recommend admission for further specialist checking. Mark as HIGH CHANCE OF CVD.")
elif initial_probability - updated_probability >= 0.15:
    print("Recommend admission to the ER for further investigation of symptoms. Mark as low probability for CVD.")

# Display model evaluation results
print("\nModel Evaluation Results:")
for model_name, result in model_results.items():
    print(f"{model_name} Accuracy: {result['accuracy']:.2%}")
    print(f"{model_name} Confusion Matrix:\n{result['confusion_matrix']}\n")

# Visualization option report
print("Visualization Option Report:")
for model_name, result in model_results.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No CVD', 'CVD'], yticklabels=['No CVD', 'CVD'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

# Printable ticket for evaluation
print("\nPrintable Ticket for Evaluation:")
print(f"Patient Information:")
for var, user_val in zip(variables, user_values):
    print(f"{var}: {user_val}")

print("\nEvaluation Results:")
print(f"Initial Probability of having CVD: {initial_probability:.2%}")
print(f"Updated Probability of having CVD: {updated_probability:.2%}")

for model_name, result in model_results.items():
    print(f"\n{model_name} Accuracy: {result['accuracy']:.2%}")
    print(f"{model_name} Confusion Matrix:\n{result['confusion_matrix']}\n")

# Check if probability increased by 10% or more
if updated_probability - initial_probability >= 0.10:
    print("Recommend admission for further specialist checking. Mark as HIGH CHANCE OF CVD.")
    # Additional actions for HIGH CHANCE OF CVD, such as marking the user or further recommendations
elif initial_probability - updated_probability >= 0.15:
    print("Recommend admission to the ER for further investigation of symptoms. Mark as low probability for CVD.")
    # Additional actions for low probability for CVD, such as marking the user or further recommendations
else:
    print("No specific recommendations at this time.")
    # Additional actions for no specific recommendations, if needed
