import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import time
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def calculate_loss(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False) # RMSE


def validate_model(model, X, y, folds=(5,5), rstate=42, verbose=True):
    """Evaluate model performance with cross validaiton

    Args:
        model (sklearn clf): model with {'fit','predict'} methods
        X,y (pandas.DataFrame or Series): features and target
        folds (tuple, optional): Do i iterations of k folds as (i,k). Defaults to (5,5).
        rstate (int, optional): CV split randomness. Defaults to 42.

    Returns:
        tuple: (mean_train, mean_valid, std_valid) losses
    """

    maxfold, num_folds = folds
    # Create a cross-validation iterator
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=rstate)

    # Initialize lists to store cross-validation results
    train_losses = []
    valid_losses = []

    # Iterate over the cross-validation folds
    for ik, (train_index, valid_index) in enumerate(kf.split(X)):

        if ik >= maxfold: break

        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        t0 = time.time()
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the training and validation data
        y_train_pred = model.predict(X_train) 
        y_valid_pred = model.predict(X_valid) 

        # Calculate and store loss scores
        train_loss = calculate_loss(y_train, y_train_pred)
        valid_loss = calculate_loss(y_valid, y_valid_pred)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        if verbose:
            print(f"CV: {ik+1}/{num_folds}, Val. loss: {valid_loss:.2f}, Time: {time.time()-t0:.1f}s")

    # Calculate the mean and standard deviation of cross-validation loss
    mean_train_loss = np.mean(train_losses)
    mean_valid_loss = np.mean(valid_losses)
    std_valid_loss = np.std(valid_losses)

    if verbose:
        # Print the results
        print(f"Mean Training loss: {mean_train_loss:.3f}")
        print(f"Mean Validation loss: {mean_valid_loss:.3f} (±{std_valid_loss:.3f})")

    return mean_train_loss, mean_valid_loss, std_valid_loss


def residual_plot(ax, train_labels, train_preds, test_labels=None, test_preds=None, 
                  title="Residual Plot", xlim=[5, 7.5]):
    """ Residual plot to evaluate performance of our simple linear regressor """
    # plt.figure(figsize=figsize)
    ax.scatter(train_preds, train_preds - train_labels, c='blue', alpha=0.1,
                marker='o', edgecolors='white', label='Training')
    
    if test_labels is not None:
        ax.scatter(test_preds, test_preds - test_labels, c='red', alpha=0.1,
                    marker='^', edgecolors='white', label='Test')
    ax.set_xlabel('Predicted values')
    ax.set_ylabel('Residuals')
    ax.hlines(y=0, xmin=xlim[0], xmax=xlim[1], color='black', lw=1)
    # ax.set_xlim(xlim)
    if test_labels is not None:
        ax.legend(loc='best')
    ax.set_title(title)
    # plt.show()
    return