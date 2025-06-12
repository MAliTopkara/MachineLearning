# KISIM 2: Model Eğitimi, Değerlendirme ve Görselleştirmeler

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# Ödev Gereksinimleri (Bu Kısımda Karşılananlar):
# 4. Pandas, Seaborn, Matplotlib vb. kütüphaneler ile gerekli görselleştirmeler (karmaşıklık matrisi tablosu, loss - epoch eğrisi vb.) yapılmalıdır.

# --- Model Eğitimi ---
print("\n--- Model Eğitimi ---")
# KISIM 1'den gelen değişkenler: X_train_scaled, y_train, X_test_scaled, y_test, target_names
# KISIM 1'den gelen model parametreleri: input_size, hidden_sizes, output_size, learning_rate

input_size = X_train_scaled.shape[1]
hidden_sizes = [10, 5] # İlk kısımda tanımlandığı gibi
output_size = len(np.unique(y_train))
learning_rate = 0.01
epochs = 1000
batch_size = 32

# Sinir ağı modelini oluştur
# Bu adım için NeuralNetwork sınıfının tanımlanmış olması gerekir.
nn_model = NeuralNetwork(input_size, hidden_sizes, output_size, learning_rate)

# Modeli eğit
training_losses = nn_model.train(X_train_scaled, y_train, epochs, batch_size)
print("Model Eğitimi Tamamlandı.")

# --- Model Değerlendirme ---
print("\n--- Model Değerlendirme ---")
y_pred_train = nn_model.predict(X_train_scaled)
y_pred_test = nn_model.predict(X_test_scaled)

# Doğruluk (Accuracy) hesapla
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Eğitim Seti Doğruluğu: {train_accuracy:.4f}")
print(f"Test Seti Doğruluğu: {test_accuracy:.4f}")

# --- Görselleştirmeler ---
print("\n--- Görselleştirmeler ---")

# 1. Kayıp (Loss) - Epoch Eğrisi
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), training_losses, label='Eğitim Kaybı')
plt.title('Kayıp (Loss) - Epoch Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Kayıp (Loss)')
plt.legend()
plt.grid(True)
plt.show()

# 2. Karmaşıklık Matrisi (Test Seti İçin)
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Karmaşıklık Matrisi (Test Seti)')
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.show()