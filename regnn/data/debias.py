import numpy as np
from sklearn.model_selection import KFold


def debias_treatment_kfold(X, D, model_class, k=5, is_classifier=False, sample_weight=None, **model_params):
    """
    Generates centered treatment residuals and returns the trained models.

    Args:
        X (np.array): Covariates matrix.
        D (np.array): Treatment vector.
        model_class: The class of the model (e.g., LogisticRegression, Ridge).
        k (int): Number of folds.
        is_classifier (bool): True for Binary D (Propensity), False for Continuous D.
        sample_weight (np.array, optional): Sample weights for model fitting. Default is None.
        **model_params: Parameters for the model.

    Returns:
        D_tilde (np.array): The residuals (D - Predicted).
        predictions (np.array): The predicted values (Propensity or E[D|X]).
        models (list): List of k trained model objects.
    """

    n_samples = X.shape[0]
    D_tilde = np.zeros(n_samples)
    predictions = np.zeros(n_samples)
    models = []  # List to store the trained models

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    print(f"Starting {k}-Fold Debiasing (Classifier={is_classifier})...")

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"{fold_i}/{k}...")
        # 1. Split Data
        X_train, D_train = X[train_idx], D[train_idx]
        X_val = X[val_idx]
        
        # Split sample weights if provided
        sample_weight_train = sample_weight[train_idx] if sample_weight is not None else None

        # 2. Instantiate new model for this fold
        model = model_class(**model_params)
        model.fit(X_train, D_train, sample_weight=sample_weight_train)

        # 3. Store the model
        models.append(model)

        # 4. Predict
        if is_classifier:
            # For Binary: Get probability of Class 1
            pred_val = model.predict_proba(X_val)[:, 1]
        else:
            # For Continuous: Get regression value
            pred_val = model.predict(X_val)

        # 5. Store Residuals
        predictions[val_idx] = pred_val
        D_tilde[val_idx] = D[val_idx] - pred_val

    return D_tilde, predictions, models
