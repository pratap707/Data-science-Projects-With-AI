import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
 
# Step 1: Load dataset
url = "https://gist.githubusercontent.com/GaneshSparkz/9dabfdeab9808d8e1b74f7fad4b91253/raw/Mall_Customers.csv"
df = pd.read_csv(url)

# Step 2: Visualize basic scatter plot
sns.scatterplot(x='Age', y='Annual Income (k$)', data=df, hue='Genre')
plt.title("Customer Segmentation by Age and Income")
plt.show()

# Step 3: Drop non-numeric/unnecessary column
df.drop('CustomerID', axis=1, inplace=True)

# Step 4: Convert 'Genre' to numeric (Male = 0, Female = 1)
df['Genre'] = df['Genre'].map({'Male': 0, 'Female': 1})

# Step 5: Feature Scaling
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

X = pd.DataFrame(scaled, columns=df.columns)

# Step 6: Elbow Method to find optimal K
inertia = []
for K in range(1, 11):
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(X)  # <- FIXED: was incorrectly using `KMeans.fit(X)` instead of `kmeans.fit(X)`
    inertia.append(kmeans.inertia_)

# Step 7: Plot Elbow Curve
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.grid(True)
#plt.show()

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

pca = PCA(n_components=2)
pca_features = pca.fit_transform(X)

df['PCA1'] = pca_features[:, 0]
df['PCA2'] = pca_features[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', hue='Cluster', data=df, palette='tab10')
plt.title("Customer Segments (PCA View)")
plt.show()