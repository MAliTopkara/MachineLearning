import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Veri setini oku
df = pd.read_csv("C:\\Users\\Monster\\Downloads\\Student_Performance.csv")

# Kategorik veriyi sayısala çevir (Yes → 1, No → 0)
df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

# Giriş ve çıkış değişkenlerini belirle
X = df[["Hours Studied", "Previous Scores", "Extracurricular Activities",
        "Sleep Hours", "Sample Question Papers Practiced"]]
y = df["Performance Index"]

# Modeli oluştur ve eğit
model = LinearRegression()
model.fit(X, y)

# Tahminler yap
y_pred = model.predict(X)

# Ortalama Kare Hata (MSE) hesapla
mse = mean_squared_error(y, y_pred)

# Sonuçları yazdır
print("Sabit Terim (intercept):", model.intercept_)
print("Katsayılar (coefficients):", model.coef_)
print("MSE (Mean Squared Error):", mse)

# -------- GÖRSELLEŞTİRME --------

# 1. Gerçek vs Tahmin Grafiği
plt.figure(figsize=(10, 5))
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Gerçek Değerler (y)")
plt.ylabel("Tahmin Edilen Değerler (ŷ)")
plt.title("Gerçek vs Tahmin Edilen Değerler (Sklearn Modeli)")
plt.grid(True)
plt.show()

# 2. Artık (residual) dağılımı
residuals = y - y_pred
plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=50, edgecolor='k')
plt.title("Tahmin Hatalarının Dağılımı (Residuals - Sklearn)")
plt.xlabel("Hata (y - ŷ)")
plt.ylabel("Frekans")
plt.grid(True)
plt.show()
