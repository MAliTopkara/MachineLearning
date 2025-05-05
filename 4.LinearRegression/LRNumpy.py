import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("C:\\Users\\Monster\\Downloads\\Student_Performance.csv")
df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

X = df[["Hours Studied", "Previous Scores", "Extracurricular Activities",
        "Sleep Hours", "Sample Question Papers Practiced"]]
y = df["Performance Index"]


X_np = np.c_[np.ones(X.shape[0]), X.values]  # bias sütunu ekle
y_np = y.values.reshape(-1, 1)

theta = np.linalg.inv(X_np.T @ X_np) @ X_np.T @ y_np  # theta = (X^T X)^-1 X^T y
y_pred = X_np @ theta
mse = np.mean((y_np - y_pred) ** 2)


print("Sabit Terim (intercept):", theta[0][0])
print("Katsayilar (coefficients):", theta[1:].flatten())
print("MSE (Mean Squared Error):", mse)

# 4. GÖRSELLEŞTİRMELER


plt.figure(figsize=(10, 5))
plt.scatter(y_np, y_pred, alpha=0.5)
plt.plot([y_np.min(), y_np.max()], [y_np.min(), y_np.max()], 'r--')
plt.xlabel("Gerçek Değerler (y)")
plt.ylabel("Tahmin Edilen Değerler (ŷ)")
plt.title("Gerçek vs Tahmin Edilen Değerler")
plt.grid(True)
plt.show()


residuals = y_np - y_pred
plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=50, edgecolor='k')
plt.title("Tahmin Hatalarının Dağılımı (Residuals)")
plt.xlabel("Hata (y - ŷ)")
plt.ylabel("Frekans")
plt.grid(True)
plt.show()
