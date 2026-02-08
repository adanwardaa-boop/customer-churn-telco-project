from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


def train_models(X, y):
    """
    Trains multiple ML models using GridSearchCV
    and returns the best model for each algorithm.
    """

    # 1. Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear'),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )
    }

    # 2. Define hyperparameter grids
    params = {
        "Logistic Regression": {
            'C': [0.01, 0.1, 1, 10]
        },
        "Random Forest": {
            'n_estimators': [100, 200]
        },
        "XGBoost": {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    }

    best_models = {}

    # 3. Train each model using GridSearchCV
    for name, model in models.items():
        print(f"Training {name}...")

        grid = GridSearchCV(
            estimator=model,
            param_grid=params[name],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )

        grid.fit(X, y)

        # Store best model
        best_models[name] = grid.best_estimator_

        print(f"Best parameters for {name}: {grid.best_params_}")
        print("-" * 50)

    # 4. Return dictionary of best models
    return best_models
