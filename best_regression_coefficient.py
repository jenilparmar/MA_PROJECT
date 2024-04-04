import numpy as np

def find_best_regression_coefficient(X_train, y_train, coefficients):
    """
    Find the best regression coefficient using grid search.

    Parameters:
    - X_train: Training features.
    - y_train: Training target.
    - coefficients: List of regression coefficients to search.

    Returns:
    - best_coefficient: Best regression coefficient found.
    """

    best_mse = float('inf')  # Initialize with a large value
    best_coefficient = None

    for coef in coefficients:
        # Fit a linear regression model with the current coefficient
        weights = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
        y_pred = X_train @ weights

        # Calculate MSE
        mse = np.mean((y_train - y_pred) ** 2)

        # Update best coefficient if current MSE is lower
        if mse < best_mse:
            best_mse = mse
            best_coefficient = coef

    return best_coefficient

# Example usage:
# Assuming X_train and y_train are your training features and target
# coefficients = [0.001, 0.01, 0.1, 1.0]  # Example coefficients to search
# best_coefficient = find_best_regression_coefficient(X_train, y_train, coefficients)
# print("Best regression coefficient:", best_coefficient)
