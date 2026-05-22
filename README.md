# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Start the program and import the required libraries for Linear Regression and Gradient Descent.
2. Load the dataset containing city population and profit values.
3. Initialize model parameters and apply the gradient descent algorithm to minimize error.
4. Train the linear regression model using the training data.
5. Predict the profit of the city using the trained model and display the results.
 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Supriya V
RegisterNumber: 212225100052 
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Load dataset
# -----------------------
data = pd.read_csv("ex3.csv")

x = data["R&D Spend"].values
y = data["Profit"].values

# -----------------------
# Feature Scaling (IMPORTANT)
# -----------------------
x = (x - np.mean(x)) / np.std(x)

# -----------------------
# Parameters
# -----------------------
w = 0.0          # weight
b = 0.0          # bias
alpha = 0.01     # learning rate
epochs = 100
n = len(x)

losses = []

# -----------------------
# Gradient Descent
# -----------------------
for i in range(epochs):
    # Prediction
    y_hat = w * x + b

    # Loss (MSE)
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    # Gradients
    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    # Update parameters
    w = w - alpha * dw
    b = b - alpha * db

# -----------------------
# Plots
# -----------------------
plt.figure(figsize=(12, 5))

# Loss vs Iterations
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

# Regression Line
plt.subplot(1, 2, 2)
plt.scatter(x, y, label="Data")
plt.plot(x, w * x + b, label="Regression Line")
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression using Gradient Descent")
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------
# Final Parameters
# -----------------------
print("Final Weight (w):", w)
print("Final Bias (b):", b)
*/
```

## Output:


<img width="598" height="296" alt="Screenshot 2026-05-22 193658" src="https://github.com/user-attachments/assets/0c2496d1-817c-4b84-82b6-e70d3bb4f23e" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
