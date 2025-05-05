# -*- coding: utf-8 -*-
"""
Part 4: Predicting House Prices using Linear Regression and k-NN Regression

This script loads the California Housing dataset, prepares it, trains
Linear Regression and k-Nearest Neighbors (k-NN) regression models,
evaluates their performance, and compares the results.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 1. Data Loading and Exploration ---
print("--- 1. Data Loading and Exploration ---")

# Load the dataset
# fetch_california_housing returns a Bunch object containing data and metadata
california_housing = fetch_california_housing(as_frame=True)

# The actual data features are in the 'data' attribute, target in 'target'
# 'frame' attribute directly provides a pandas DataFrame
dataset_df = california_housing.frame

# Display dataset information
print("Dataset Description:")
print(california_housing.DESCR[:500], "...\n") # Print first 500 chars of description

print("Dataset Head:")
print(dataset_df.head())
print("\nDataset Info:")
dataset_df.info()
print("\nDataset Statistical Summary:")
print(dataset_df.describe())

# Define features (X) and target (y)
features = california_housing.feature_names
target_variable = california_housing.target_names[0] # Usually 'MedHouseVal'

X = dataset_df[features]
y = dataset_df[target_variable]

print(f"\nFeatures (X shape): {X.shape}")
print(f"Target (y shape): {y.shape}")
print("-" * 30, "\n")


# --- 2. Data Preparation ---
print("--- 2. Data Preparation ---")

# Split the dataset into training (80%) and testing (20%) sets
# random_state ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing set size: X_test={X_test.shape}, y_test={y_test.shape}")

# Standardize features (scaling)
# StandardScaler removes the mean and scales to unit variance
# It's important to fit the scaler ONLY on the training data
scaler = StandardScaler()

# Fit on training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# X_train_scaled and X_test_scaled are now NumPy arrays
print("\nFeatures have been scaled using StandardScaler.")
print(f"Example scaled training data (first row):\n{X_train_scaled[0]}")
print("-" * 30, "\n")


# --- 3. Model Training ---
print("--- 3. Model Training ---")

# --- Linear Regression Model ---
# Initialize the Linear Regression model
linear_reg_model = LinearRegression()

# Train the model using the scaled training data
# .fit() learns the relationship between X_train_scaled and y_train
linear_reg_model.fit(X_train_scaled, y_train)
print("Linear Regression model trained successfully.")

# --- k-Nearest Neighbors (k-NN) Regression Models ---
# Define the different values of k (number of neighbors) to try
k_values = [3, 5, 7]
knn_models = {} # Dictionary to store trained k-NN models

# Loop through each k value
for k in k_values:
    # Initialize the k-NN Regressor model with the current k
    knn_model = KNeighborsRegressor(n_neighbors=k)

    # Train the model using the scaled training data
    knn_model.fit(X_train_scaled, y_train)

    # Store the trained model
    knn_models[k] = knn_model
    print(f"k-NN Regression model with k={k} trained successfully.")

print("-" * 30, "\n")


# --- 4. Model Evaluation ---
print("--- 4. Model Evaluation ---")

# Dictionary to store evaluation results
model_performance = {}

# --- Evaluate Linear Regression ---
# Make predictions on the scaled test data
y_pred_lr = linear_reg_model.predict(X_test_scaled)

# Calculate performance metrics
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)

# Store metrics
model_performance['Linear Regression'] = {'MAE': lr_mae, 'MSE': lr_mse, 'R2': lr_r2}
print("Linear Regression Evaluation:")
print(f"  Mean Absolute Error (MAE): {lr_mae:.4f}")
print(f"  Mean Squared Error (MSE): {lr_mse:.4f}")
print(f"  R-squared (R2): {lr_r2:.4f}\n")

# --- Evaluate k-NN Models ---
# Loop through the trained k-NN models
for k, model in knn_models.items():
    # Make predictions on the scaled test data
    y_pred_knn = model.predict(X_test_scaled)

    # Calculate performance metrics
    knn_mae = mean_absolute_error(y_test, y_pred_knn)
    knn_mse = mean_squared_error(y_test, y_pred_knn)
    knn_r2 = r2_score(y_test, y_pred_knn)

    # Store metrics
    model_performance[f'k-NN (k={k})'] = {'MAE': knn_mae, 'MSE': knn_mse, 'R2': knn_r2}
    print(f"k-NN (k={k}) Evaluation:")
    print(f"  Mean Absolute Error (MAE): {knn_mae:.4f}")
    print(f"  Mean Squared Error (MSE): {knn_mse:.4f}")
    print(f"  R-squared (R2): {knn_r2:.4f}\n")

print("-" * 30, "\n")


# --- 5. Performance Comparison and Interpretation ---
print("--- 5. Performance Comparison and Interpretation ---")

# Create a DataFrame for easy comparison of metrics
performance_df = pd.DataFrame(model_performance).T # Transpose for better readability
performance_df = performance_df[['R2', 'MAE', 'MSE']] # Reorder columns

print("Side-by-side Model Performance Metrics:")
print(performance_df)

print("\nInterpretation:")
print("-" * 15)

# Find best model based on R2 (higher is better) and MAE (lower is better)
best_r2_model = performance_df['R2'].idxmax()
best_mae_model = performance_df['MAE'].idxmin()

print(f"Best Model based on R-squared: {best_r2_model} (R2 = {performance_df.loc[best_r2_model, 'R2']:.4f})")
print(f"Best Model based on Mean Absolute Error: {best_mae_model} (MAE = {performance_df.loc[best_mae_model, 'MAE']:.4f})")

print("\nDiscussion:")
print("1. Performance Metrics:")
print("   - R-squared (R2): Represents the proportion of variance in the target variable explained by the model. Higher is better (max 1.0).")
print("   - Mean Absolute Error (MAE): Average absolute difference between predicted and actual values. Lower is better. Units are the same as the target variable.")
print("   - Mean Squared Error (MSE): Average of the squared differences. Penalizes larger errors more heavily. Lower is better.")
print("\n2. Model Comparison:")
print("   - Linear Regression assumes a linear relationship between features and the target. It's simple, fast, and interpretable (coefficients show feature importance/direction). Its performance here provides a baseline.")
print("   - k-NN Regression is non-parametric, making no assumption about the data distribution. It predicts based on the average target value of its 'k' nearest neighbors in the feature space.")
print("   - k-NN's performance is sensitive to the choice of 'k' and requires feature scaling (which we did). Different values of 'k' yield different results.")
print("   - Comparing the metrics (R2, MAE, MSE), we can see which model (Linear Regression or one of the k-NN variants) fits the test data better for this specific dataset.")
# You would add specific observations here based on the printed performance_df
# For example: "The k-NN model with k=7 showed the highest R2 score, suggesting..."
# Or: "Linear Regression had a lower MAE than k-NN with k=3, indicating..."

print("\n3. Interpretability & Use Case:")
print("   - Linear Regression is more interpretable due to its coefficients.")
print("   - k-NN is less interpretable ('black box') but can capture complex, non-linear patterns if they exist.")
print("   - The choice between them depends on whether interpretability or potentially higher accuracy (if k-NN performs better) is prioritized.")

print("\n--- End of Analysis ---")