import shap
def shap_analysis(model, X, feature_names):
    explainer=shap.Explainer(model, X)
    shap_values=explainer(X)
    shap.summary_plot(shap_values, X, feature_names=feature_names)