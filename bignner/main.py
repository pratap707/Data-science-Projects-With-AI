# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


'''# Variables and Types
x = 10
y = 3.5
text = "Hello, Data Science!"
print(x, y, text)

# Lists and Loops
my_list = [1, 2, 3, 4]
for num in my_list:
    print(num ** 2)

# Function
def square(n):
    return n * n

print(square(5))

# Create NumPy Array
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])

# Array Operations
print(a + 10)
print(b.mean(), b.shape)

'''

# Load Dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# View First Rows
print(df.head())

df_clean = df.dropna()

df_clean["Sex"] = df_clean["Sex"].map({"male": 0, "female": 1})

df_clean= df_clean.drop(["Name", "Ticket","Cabin"], axis=1)
df_clean.head() 

sns.countplot(x="Survived", data=df)

sns.boxplot(x="Pclass", y="Age", data=df)
#plt.show()

print(df["Age"].describe())

corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()