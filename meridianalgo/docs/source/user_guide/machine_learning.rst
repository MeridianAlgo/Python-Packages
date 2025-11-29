.. _machine_learning:

Machine Learning
================

This guide covers machine learning capabilities in MeridianAlgo for financial forecasting, pattern recognition, and algorithmic trading.

Introduction
-----------

Machine learning can help identify complex patterns in financial data that traditional methods might miss. MeridianAlgo provides tools for feature engineering, model training, and evaluation.

Feature Engineering
-----------------

Good features are crucial for successful ML models:

.. code-block:: python

    import meridianalgo as ma
    import pandas as pd
    import numpy as np
    
    # Get price data
    symbol = 'AAPL'
    data = ma.get_market_data([symbol], start_date='2019-01-01', end_date='2021-01-01')
    prices = data[symbol]
    
    # Initialize feature engineer
    fe = ma.FeatureEngineer()
    
    # Create technical features
    features = fe.create_technical_features(prices)
    
    # Create statistical features
    features = fe.create_statistical_features(prices, features)
    
    # Create lag features
    features = fe.create_lag_features(prices, features, lags=[1, 5, 10])
    
    # Create rolling window features
    features = fe.create_rolling_features(prices, features, windows=[5, 10, 20])
    
    print("Feature columns:", features.columns.tolist())
    print(features.head())

Time Series Forecasting
-----------------------

### LSTM for Price Prediction

Long Short-Term Memory (LSTM) networks are effective for time series forecasting:

.. code-block:: python

    # Prepare data for LSTM
    returns = prices.pct_change().dropna()
    
    # Create features and target
    X, y = fe.prepare_lstm_data(returns, sequence_length=20, prediction_horizon=1)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Initialize and train LSTM
    lstm = ma.LSTMPredictor(
        sequence_length=20,
        hidden_size=50,
        num_layers=2,
        epochs=50,
        batch_size=32
    )
    
    lstm.fit(X_train, y_train)
    
    # Make predictions
    predictions = lstm.predict(X_test)
    
    # Evaluate
    mse = np.mean((y_test - predictions) ** 2)
    print(f"MSE: {mse:.6f}")
    
    # Plot predictions vs actual
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Returns', alpha=0.7)
    plt.plot(predictions, label='Predicted Returns', alpha=0.7)
    plt.title('LSTM Predictions vs Actual Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

### Random Forest for Direction Prediction

Predict price direction (up/down):

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    
    # Create binary target (1 if price goes up, 0 if down)
    target = (returns.shift(-1) > 0).astype(int)
    
    # Align features and target
    features_clean = features.iloc[:-1]  # Remove last row (no future return)
    target_clean = target.iloc[:-1]
    
    # Remove any NaN values
    mask = ~(features_clean.isnull().any(axis=1) | target_clean.isnull())
    X = features_clean[mask]
    y = target_clean[mask]
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))

Model Evaluation
---------------

Proper model evaluation is crucial:

.. code-block:: python

    # Cross-validation for time series
    evaluator = ma.ModelEvaluator()
    
    # Time series cross-validation
    cv_results = evaluator.time_series_cv(
        model=rf,
        X=X.values,
        y=y.values,
        n_splits=5
    )
    
    print("Cross-validation results:")
    for metric, values in cv_results.items():
        print(f"{metric}: {np.mean(values):.3f} ± {np.std(values):.3f}")

Ensemble Methods
---------------

Combine multiple models for better predictions:

.. code-block:: python

    class EnsemblePredictor:
        def __init__(self, models):
            self.models = models
            
        def predict(self, X):
            predictions = []
            for model in self.models:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)[:, 1]  # Probability of class 1
                else:
                    pred = model.predict(X)
                predictions.append(pred)
            return np.mean(predictions, axis=0)
    
    # Create ensemble
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    models = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        LogisticRegression(random_state=42),
        SVC(probability=True, random_state=42)
    ]
    
    ensemble = EnsemblePredictor(models)
    
    # Train each model
    for model in models:
        model.fit(X_train, y_train)
    
    # Ensemble prediction
    ensemble_pred = ensemble.predict(X_test)
    ensemble_binary = (ensemble_pred > 0.5).astype(int)
    
    ensemble_accuracy = accuracy_score(y_test, ensemble_binary)
    print(f"Ensemble Accuracy: {ensemble_accuracy:.3f}")

Hyperparameter Tuning
--------------------

Optimize model parameters:

.. code-block:: python

    from sklearn.model_selection import GridSearchCV
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # Grid search
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

Model Interpretation
-------------------

Understand what your model is learning:

.. code-block:: python

    import shap
    
    # SHAP values for model interpretation
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    
    # Plot feature importance
    shap.summary_plot(shap_values[1], X_test, plot_type="bar")
    
    # Plot SHAP values for a single prediction
    shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test.iloc[0])

Best Practices
-------------

1. Always use a proper train/validation/test split
2. Consider the time series nature of financial data
3. Use cross-validation carefully to avoid look-ahead bias
4. Regularly retrain models with new data
5. Combine multiple models for robustness
6. Always include a baseline model for comparison

Next Steps
----------

- Learn about :ref:`backtesting` to test your ML strategies
- Explore :ref:`portfolio_optimization` with ML predictions
- Check the :ref:`api_reference` for all ML functions
