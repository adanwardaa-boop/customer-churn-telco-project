from sklearn.metrics import classification_report , roc_auc_score
def evaluate_model(model,x,y):
    preds=model.predict(x)
    probs=model.predict_proba(x)[:,1]
    print(classification_report(y,preds))
    print("ROC-AUC:", roc_auc_score(y, probs))