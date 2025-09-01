import pandas as pd


df= pd.read_csv("creditcard.csv")


#print(df.shape)
#print(df.head)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X= df.drop("Class", axis=1)
y= df["Class"]


import numpy as np

# Replace inf/-inf with nan
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaNs
X.dropna(inplace=True)
y = y.loc[X.index]



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

rf  = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_resampled, y_resampled)

import xgboost as xgb

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_resampled, y_resampled)

from sklearn.metrics import classification_report, roc_auc_score

y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

print("Random Forest:\n", classification_report(y_test, y_pred_rf))
print("XGBoost:\n", classification_report(y_test, y_pred_xgb))

#print("ROC AUC RF:", roc_auc_score(y_test, rf.predict_log_proba(X_test)[:,1]))
print("ROC AUC XGB:", roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1]))
print("ROC AUC RF:", roc_auc_score(y_test, rf_probs))
import shap

explainer = shap.Explainer(xgb_model, X_test)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test)

import joblib

joblib.dump(xgb_model, "fraud_detector.pkl")
joblib.dump(scaler, "scaler.pkl")