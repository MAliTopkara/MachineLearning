import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import time

df = pd.read_csv("C:\\Users\\Monster\\OneDrive\\Desktop\\logisticregressiondataset.csv")

df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean())

df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})  # 1: hasta, 0: sağlıklı

X = pd.get_dummies(df.drop("Dataset", axis=1), drop_first=True)
y = df["Dataset"].values.reshape(-1, 1)

X = (X - X.mean()) / X.std()

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42
)

m, n = X_train.shape
weights = np.zeros((n, 1))
bias = 0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

learning_rate = 0.01
epochs = 1000
loss_list = []

start_train = time.time()

for i in range(epochs):
    z = np.dot(X_train, weights) + bias
    y_pred = sigmoid(z)

    loss = -np.mean(y_train * np.log(y_pred + 1e-8) + (1 - y_train) * np.log(1 - y_pred + 1e-8))
    loss_list.append(loss)

    dw = np.dot(X_train.T, (y_pred - y_train)) / m
    db = np.sum(y_pred - y_train) / m

    weights -= learning_rate * dw
    bias -= learning_rate * db

    if i % 100 == 0:
        print(f"Epoch {i} - Loss: {loss:.4f}")

end_train = time.time()
training_time = end_train - start_train
print(f"\n Eğitim süresi: {training_time:.4f} saniye")

start_test = time.time()

z_test = np.dot(X_test, weights) + bias
y_pred_test = sigmoid(z_test)
y_pred_label = (y_pred_test > 0.5).astype(int)

end_test = time.time()
testing_time = end_test - start_test
print(f" Test süresi: {testing_time:.6f} saniye")

accuracy = np.mean(y_pred_label == y_test)
print(f"\n Test doğruluğu: {accuracy:.4f}")

plt.plot(loss_list)
plt.title("Eğitim Süresince Kayıp (Loss) Grafiği")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

cm = confusion_matrix(y_test, y_pred_label)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Karışıklık Matrisi (Test Verisi)")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.show()


