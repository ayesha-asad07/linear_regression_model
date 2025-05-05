# California House Price Prediction using Regression Models

## Overview

This project demonstrates how to predict median house prices in California using the standard California Housing dataset available from Scikit-learn. It implements, trains, evaluates, and compares two fundamental regression algorithms:

1.  **Linear Regression:** A parametric model assuming a linear relationship between features and the target variable.
2.  **k-Nearest Neighbors (k-NN) Regression:** A non-parametric model that predicts based on the average value of the nearest neighbors in the feature space.

The goal is to showcase a basic machine learning workflow including data loading, preparation (splitting, scaling), model training, evaluation using common metrics (R², MAE, MSE), and performance comparison.

## Files

*   `Regression_Model.py`: The main Python script containing all the code for data loading, preprocessing, model training, evaluation, and comparison. 
*   `requirements.txt`: A file listing the necessary Python libraries for this project.

## Requirements

To run this project, you need Python 3 and the libraries listed in `requirements.txt`. You can install them using pip:

## How to Run

1.  **Clone the repository or download the files.**
2.  **Navigate to the project directory** in your terminal or command prompt.
3.  **Install the required packages** (if you haven't already):
    ```bash
    pip install -r requirements.txt
    ```
4.  **Execute the Python script:**
    ```bash
    python Regression_Model.py
    ```

## Expected Output

Running the script will print information directly to your console, including:

*   Confirmation of data loading and preparation steps.
*   Status updates on model training.
*   Performance metrics (R-squared, Mean Absolute Error, Mean Squared Error) for the Linear Regression model on the test set.
*   Performance metrics for the k-NN Regression model (tested with k=3, 5, and 7) on the test set.
*   A final side-by-side comparison table of the metrics for all models.
*   A brief discussion interpreting the results and comparing the models.


