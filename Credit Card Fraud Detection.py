import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report,roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve


#print("Path to dataset files:", path)
df = pd.read_csv("creditcard.csv")
print("Dataset loaded successfully!")
print(df.head())
