import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load dataset
url = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"
df = pd.read_csv(url)

# Visualizations
sns.countplot(x="target", hue="sex", data=df)
plt.title("Heart Disease Distribution")
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Features and Target
X = df.drop("target", axis=1)
Y = df["target"]

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Handle imbalance
smote = SMOTE(random_state=42)
X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train_scaled, Y_train_smote)
Y_pred_log = log_model.predict(X_test_scaled)

print("Logistic Regression Result:")
print(classification_report(Y_test, Y_pred_log))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_smote, Y_train_smote)
Y_pred_rf = rf_model.predict(X_test)

print("Random Forest Result:")
print(classification_report(Y_test, Y_pred_rf))

sns.heatmap(confusion_matrix(Y_test, Y_pred_rf), annot=True, fmt="d")
plt.title("Confusion Matrix - Random Forest")
plt.show()
