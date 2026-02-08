
import sys
import os

# ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import engineer_features
from src.train_models import train_models
from src.evaluate import evaluate_model
from src.interpretability import shap_analysis
import joblib

def main():
    df = load_and_clean_data("data/raw/telco_churn.csv")
    X, y, feature_names = engineer_features(df)

    models = train_models(X, y)
    best_model = models["XGBoost"]

    evaluate_model(best_model, X, y)
    joblib.dump(best_model, "best_model.pkl")

    shap_analysis(best_model, X, feature_names)

if __name__ == "__main__":
    main()
