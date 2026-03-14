import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("ecommerce_sales_data.csv")

# Use Quantity as feature and Sales as target
X = df[["Quantity"]].values.astype(float)
y = df["Sales"].values.astype(float).reshape(-1, 1)

m = len(y)

# Feature normalization
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Add bias term
X = np.c_[np.ones(m), X]

# Initialize parameters
theta = np.zeros((2, 1))
learning_rate = 0.01
iterations = 1000

def compute_cost(X, y, theta):
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

def gradient_descent(X, y, theta, lr, iterations):
    cost_history = []

    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y

        gradients = (1 / m) * X.T.dot(errors)
        theta = theta - lr * gradients

        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")

    return theta, cost_history

theta_final, costs = gradient_descent(X, y, theta, learning_rate, iterations)

print("\nFinal Parameters:")
print("Theta0 (Intercept):", round(theta_final[0][0], 4))
print("Theta1 (Slope for Quantity):", round(theta_final[1][0], 4))

# Plot cost vs iterations
plt.plot(range(iterations), costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations (Gradient Descent)")
plt.savefig("cost_curve.png")
plt.show()

# Predictions
y_pred = X.dot(theta_final)

# Plot actual vs predicted
plt.scatter(y, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.savefig("actual_vs_predicted.png")
plt.show()
