'''import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split

df = pd.DataFrame({
    'age':[25, 40,52,33,45],
    'income':[30000, 70000,120000,450000,900000],
    'gender':['Male','Female','Female','Male','Male'],
    'target':[0,1,0,0,1]
})

df['age_group'] = pd.cut(df['age'], bins=[0,30,45,60], labels=["Young","Mddle","Senior"])

df = pd.get_dummies(df, columns=['gender','age_group'], drop_first=True)

df['income_log'] = np.log1p(df['income'])

df['age_income'] = df['age'] * df['income_log']

print(df)

from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score,roc_curve,precision_recall_curve
import matplotlib.pyplot as plt 
import numpy as np

y_true =  [0]*90 + [1]*10
y_probs = np.random.rand(100)
y_pred = [1 if p > 0.5 else 0 for p in y_probs]

# confusion metrix
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

#classification report(precision, recall, f1)
print("\n Classification Report:")
print(classification_report(y_true, y_pred))

#ROC-AUC
roc_auc = roc_auc_score(y_true, y_probs)
print(f"🔹 ROC AUC Score: {roc_auc:.2f}")

fpr, tpr, _ = roc_curve(y_true, y_probs)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC={roc_auc:.2f})")
plt.plot([0,1], [0,1], linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()
#Random Forest (Bagging)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

#create dummy data
X, Y = make_classification(n_samples=1000, n_features=10, n_classes=2, weights=[0.9, 0.1], random_state=42)

#split
X_train ,X_test, Y_train, Y_test = train_test_split(X,Y, stratify=Y, test_size=0.2, random_state=42)

#train random forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train,Y_train)

#predict 
Y_pred = rf.predict(X_test)

#evaluate
#print(classification_report(Y_test, Y_pred))

# XGBoost (Boosting)
from xgboost import XGBClassifier

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, Y_train)
Y_pred_xgb = xgb.predict(X_test)

#print(classification_report(Y_test, Y_pred_xgb))

#stacking
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
] 

stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stack.fit(X_train, Y_train)

Y_pred_stack = stack.predict(X_test)
print(classification_report(Y_test, Y_pred_stack))

#full search of combination (random forest)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define hyperparameter grid (fixed typo in 'min_samples_split')
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],  
}

# GridSearchCV setup with random forest
grid = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1
)

# Make sure X_train and y_train are defined before this point
grid.fit(X_train, y_train)


# Output best model
print("Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_

#Random Search – Faster & Effective

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': [5, 10, None],
    'min_samples_split': randint(2, 10),
}

random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist,
                                   n_iter=10, cv=3, scoring='f1', n_jobs=1, random_state=42)
random_search.fit(X_train, Y_train)

print("Best Parameters:", random_search.best_params_)
best_rf = random_search.best_estimator_
#Optuna (Auto ML Style)

import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    return cross_val_score(clf, X_train, Y_train, cv=3, scoring='f1').mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best Trial:", study.best_trail.params)

#SMOTE + RandomForest(imbalnce)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#simulate imbalanced data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=5000, weights=[0.95, 0.05], random_state=42)

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
 #apply smote
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
# train model
model = RandomForestClassifier()
model.fit(X_res, y_res)
y_pred = model.predict(X_test)
#evaluate
#print(classification_report(y_test, y_pred)) 

#SMOTE + Tomek Links

from imblearn.combine import SMOTETomek

resample = SMOTETomek(random_state=42)
X_res, y_res = resample.fit_resample(X_train, y_train)

#Pipeline + StratifiedKFold


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.datasets import make_classification
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

#simulated data
X, y = make_classification(n_samples=1000, weights=[0.9,0.1], random_state=42)

#define pipeline
#pipe = Pipeline([
imb_pipe = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    
])

#cross-validation 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#scores = cross_val_score(pipe, X, y, scoring='f1', cv=cv)
scores = cross_val_score(imb_pipe, X, y, scoring='f1', cv=cv)


print("F1 Scores:", scores)
print("Average F1 Score:", scores.mean())
print("F1 Score with SMOTE Pipeline:", scores.mean)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
model = RandomForestClassifier().fit(X, y)

importances = model.feature_importances_
features = [f'Features {i}' for i in range(X.shape[1])]

plt.barh(features, importances)
plt.title("Features ")
plt.xlabel("Importance")
plt.show()


import shap
import xgboost as xgb

model = xgb.XGBClassifier(use_label_encoder= False, eval_metric='logloss')
model.fit(x, y)

explainer = shap.Explainer(model, x)
shap_values = explainer(X)

shap.summary_plot(shap_values, x)
import pandas as pd
import shap
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Step 1: Load and preprocess dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)
df.dropna(inplace=True)

# Encode categorical features
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

# Features and Target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Step 2: Train XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

# Step 3: SHAP explainability
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Step 4: Summary plot (opens in a browser or inline if Jupyter)
#shap.summary_plot(shap_values, X)'''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Step 1: Load and preprocess dataset (e.g., Titanic)
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)
df.dropna(inplace=True)

# Encode categorical columns
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

# Step 2: Features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 4: Train RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Save model using joblib
joblib.dump(model, "fraud_model.pkl")
print("✅ Model saved as fraud_model.pkl")

loaded_model = joblib.load("fraud_model.pkl")

pred = loaded_model.predict(X_test)
print("Prediction:", pred[:5])