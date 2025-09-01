import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# === 📊 Visualizations ===
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
#plt.show()

sns.boxplot(x='Pclass', y='Age', data=df)
plt.title("Age Distribution by Passenger Class")
#plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
#plt.show()

# === 🔧 Data Preprocessing ===
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Fill missing Age values
df['Age'] = df['Age'].fillna(df['Age'].median())

# Drop Cabin (too many missing), Ticket and Name (not useful)
df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

# Drop missing rows from 'Embarked' and 'Fare' if any left
df.dropna(inplace=True)

# Encode 'Sex' (Label Encoding: male=1, female=0)
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

# One-hot encode 'Embarked' (and drop the first column to avoid dummy trap)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Standard scaling for Age and Fare
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Print the cleaned dataset
#print(df.head())
#Supervised Learning Models (Classification)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = df[['Pclass','Sex','Age','Fare']]
Y = df['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

#print("Accuracy:", accuracy_score(Y_test, Y_pred))

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
#print(confusion_matrix(Y_test, Y_pred))
#print(classification_report(Y_test, Y_pred))

from sklearn.model_selection import cross_val_score

scores = cross_val_score(LogisticRegression(), X, Y, cv=5)
#print("Cross-validation scores:", scores)
#print("Average accuracy:", scores.mean)

#Unsupervised Learning (Clustering + PCA)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
#print(kmeans.labels_)

#PCA
pca =PCA(n_components=2)
X_pca = pca.fit_transform(X)

#Pipelines and ColumnTransformer

from sklearn.pipeline import  Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

numeric = ['Age', 'Fare']
categorical =['Sex']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric),
    ('cat', OneHotEncoder(),categorical)
])

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

pipe.fit(X_train, Y_train)
pipe.score(X_test, Y_test)

#Handling Imbalanced Datasets

from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_res, Y_res = smote.fit_resample(X_train, Y_train)

print(df.head())
print(df.tail())