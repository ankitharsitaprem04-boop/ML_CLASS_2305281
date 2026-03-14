import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Load data
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to calculate mean and variance per feature per class
def summarize_by_class(X, y):
    summaries = {}
    for class_value in np.unique(y):
        X_class = X[y == class_value]
        summaries[class_value] = [(np.mean(col), np.var(col)) for col in X_class.T]
    return summaries

# Function to calculate Gaussian probability
def gaussian_prob(x, mean, var):
    eps = 1e-6  # to avoid division by zero
    exponent = np.exp(-(x - mean)**2 / (2 * var + eps))
    return (1 / np.sqrt(2 * np.pi * var + eps)) * exponent

# Function to calculate class probabilities
def calculate_class_probs(summaries, X_row, class_priors):
    probs = {}
    for class_value, features in summaries.items():
        probs[class_value] = class_priors[class_value]
        for i in range(len(features)):
            mean, var = features[i]
            probs[class_value] *= gaussian_prob(X_row[i], mean, var)
    return probs

# Function to predict
def predict(summaries, X, y):
    predictions = []
    class_priors = {c: np.sum(y==c)/len(y) for c in np.unique(y)}
    for row in X:
        probs = calculate_class_probs(summaries, row, class_priors)
        predictions.append(max(probs, key=probs.get))
    return predictions

# Train
summaries = summarize_by_class(X_train, y_train)

# Predict
y_pred = predict(summaries, X_test, y_train)

# Evaluate
accuracy = np.mean(y_pred == y_test)
print("Manual Gaussian NB Accuracy:", accuracy)
