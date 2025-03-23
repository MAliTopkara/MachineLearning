import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("C:\\Users\\Monster\\OneDrive\\Desktop\\logisticregressiondataset.csv")

df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean())

df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})

X = df.drop("Dataset", axis=1)
y = df["Dataset"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)

start_train = time.time()
model.fit(X_train, y_train)
end_train = time.time()
training_time = end_train - start_train
print(f"\n Eğitim süresi: {training_time:.4f} saniye")

start_test = time.time()
y_pred = model.predict(X_test)
end_test = time.time()
testing_time = end_test - start_test
print(f"Test süresi: {testing_time:.6f} saniye")

print("\nModel Başarı Değerlendirmesi:")
print("Doğruluk (Accuracy):", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.show()
