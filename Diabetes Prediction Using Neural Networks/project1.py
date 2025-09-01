import pandas as pd
import numpy as np

np.random.seed(42)


n_samples = 1000
data = pd.DataFrame({
    'age': np.random.randint(18, 65, n_samples),
    'income': np.random.normal(50000, 15000, n_samples),
    'debt': np.random.normal(15000, 5000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'loan_amount': np.random.normal(10000, 3000, n_samples),
})

data ['risk'] = ((data['debt'] > 20000) | (data['credit_score'] < 600)).astype(int)


data.to_csv('financial_risk_data.csv', index=False)
print("Dataset created and saved as financial_risk_data.csv")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("financial_risk_data.csv")

X = data.drop('risk', axis=1)
y = data['risk']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, test_size=0.2, random_state=42)

import tensorflow as tf
from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import control_dependencies
from tensorflow.keras.layers import Dense, Dropout, Input

# define the model
model = Sequential([
    Dense(16, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
]) 

#compile the model
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

# train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n Test Accuracy: {accuracy:.2f}")

y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)
print(classification_report(y_test, y_pred_classes))


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
print(plt.show())


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# define a better model
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

# callbacks 


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

# train the model 
model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint]
    
)

model.load_weights('best_model.h5')

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n Test Accuracy: {accuracy:.2f}")

y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred_class))

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
print(plt.legend())
print(plt.show())


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()