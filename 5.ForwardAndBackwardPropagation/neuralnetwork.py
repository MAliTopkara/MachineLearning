# KISIM 1: Sinir Ağı Modeli Tanımı ve Veri Hazırlığı

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Ödev Gereksinimleri (Bu Kısımda Karşılananlar):
# 1. NumPy, Pandas vb. yardımcı kütüphaneler kullanılabilir ancak model class olarak yazılmalıdır. Modeller için hazır kütüphaneler kullanılmamalıdır.
# 2. Problem veri setine bağlı olarak herhangi bir problem (sınıflandırma, regresyon vb.) seçilebilir. Uygun bir veri seti seçilmelidir.
# 3. İleri yayılım, geri yayılım, aktivasyon fonksiyonları, gradyan iniş algoritması vb. gerekli birimler uygulanmalıdır.

class NeuralNetwork:
    """
    İleri ve geri yayılım algoritmalarını kullanarak eğitilebilen basit bir sinir ağı modeli.
    """

    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        """
        Sinir ağının katmanlarını, ağırlıklarını ve bias'larını başlatır.

        Args:
            input_size (int): Giriş katmanındaki nöron sayısı.
            hidden_sizes (list): Her gizli katmandaki nöron sayılarının listesi.
            output_size (int): Çıkış katmanındaki nöron sayısı.
            learning_rate (float): Ağırlık güncellemeleri için öğrenme oranı.
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights = {}
        self.biases = {}
        self.activations = {}  # Her katmanın aktivasyon çıkışlarını saklayacak (a)
        self.z_values = {}  # Her katmanın aktivasyon öncesi değerlerini (weighted sum) saklayacak (z)

        # Ağırlıkları ve bias'ları başlatma
        # Giriş katmanı ile ilk gizli katman arasındaki ağırlıklar ve bias'lar
        if len(self.hidden_sizes) == 0:  # Gizli katman yoksa, doğrudan girişten çıkışa
            self.weights['W1'] = np.random.randn(self.input_size, self.output_size) * 0.01
            self.biases['b1'] = np.zeros((1, self.output_size))
        else:
            self.weights['W1'] = np.random.randn(self.input_size, self.hidden_sizes[0]) * 0.01
            self.biases['b1'] = np.zeros((1, self.hidden_sizes[0]))

            # Diğer gizli katmanlar arası ağırlıklar ve bias'lar
            for i in range(len(self.hidden_sizes) - 1):
                self.weights[f'W{i + 2}'] = np.random.randn(self.hidden_sizes[i], self.hidden_sizes[i + 1]) * 0.01
                self.biases[f'b{i + 2}'] = np.zeros((1, self.hidden_sizes[i + 1]))

            # Son gizli katman ile çıkış katmanı arasındaki ağırlıklar ve bias'lar
            self.weights[f'W{len(self.hidden_sizes) + 1}'] = np.random.randn(self.hidden_sizes[-1],
                                                                             self.output_size) * 0.01
            self.biases[f'b{len(self.hidden_sizes) + 1}'] = np.zeros((1, self.output_size))

    # Aktivasyon Fonksiyonları ve Türevleri
    def relu(self, x):
        """ReLU aktivasyon fonksiyonu."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """ReLU aktivasyon fonksiyonunun türevi."""
        return (x > 0).astype(float)

    def softmax(self, x):
        """Softmax aktivasyon fonksiyonu (çok sınıflı sınıflandırma için)."""
        # Sayısal kararlılık için maksimum değeri çıkarılır
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward_propagation(self, X):
        """
        İleri yayılım algoritması. Giriş verisini ağ üzerinden yayarak çıkış tahmini yapar.

        Args:
            X (np.array): Giriş verisi.

        Returns:
            np.array: Çıkış katmanının aktivasyonları (tahmin edilen olasılıklar).
        """
        self.activations['a0'] = X  # Giriş katmanı aktivasyonu

        num_layers = len(self.hidden_sizes)

        # Gizli Katmanlar için ileri yayılım
        for i in range(num_layers):
            layer_input = self.activations[f'a{i}']
            W = self.weights[f'W{i + 1}']
            b = self.biases[f'b{i + 1}']

            self.z_values[f'z{i + 1}'] = np.dot(layer_input, W) + b
            self.activations[f'a{i + 1}'] = self.relu(self.z_values[f'z{i + 1}'])

        # Çıkış Katmanı için ileri yayılım
        output_layer_input = self.activations[f'a{num_layers}']
        W_out = self.weights[f'W{num_layers + 1}']
        b_out = self.biases[f'b{num_layers + 1}']

        self.z_values[f'z{num_layers + 1}'] = np.dot(output_layer_input, W_out) + b_out
        self.activations[f'a{num_layers + 1}'] = self.softmax(self.z_values[f'z{num_layers + 1}'])

        return self.activations[f'a{num_layers + 1}']

    def cross_entropy_loss(self, y_true, y_pred):
        """
        Çapraz Entropi Kayıp Fonksiyonu (çok sınıflı sınıflandırma için).
        """
        m = y_true.shape[0]
        # Logaritmanın sıfır olmaması için çok küçük bir değer ekleyelim
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)  # Sayısal kararlılık için

        # y_true'yu one-hot encoded formata çevirelim
        y_true_one_hot = np.zeros_like(y_pred)
        y_true_one_hot[np.arange(m), y_true] = 1

        loss = -np.sum(y_true_one_hot * np.log(y_pred)) / m
        return loss

    def backward_propagation(self, X, y, output):
        """
        Geri yayılım algoritması. Hata gradyanlarını hesaplar ve ağırlık/bias güncellemelerini belirler.

        Args:
            X (np.array): Giriş verisi (batch).
            y (np.array): Gerçek etiketler (batch).
            output (np.array): İleri yayılımdan elde edilen çıkış tahminleri.
        """
        gradients = {}
        num_layers = len(self.hidden_sizes)

        # y'yi one-hot encoded formatına dönüştürelim
        y_one_hot = np.zeros_like(output)
        y_one_hot[np.arange(len(y)), y] = 1

        # Çıkış katmanı hatası (softmax ve çapraz entropi türevi için basitleştirilmiş)
        delta = output - y_one_hot  # (N, output_size)

        # Çıkış katmanı ağırlık ve bias gradyanları
        gradients[f'dW{num_layers + 1}'] = np.dot(self.activations[f'a{num_layers}'].T, delta)
        gradients[f'db{num_layers + 1}'] = np.sum(delta, axis=0, keepdims=True)

        # Gizli katmanlar için hata yayılımı ve gradyan hesaplaması (sondan başa doğru)
        for i in range(num_layers, 0, -1):
            # Bir önceki katmana yayılacak hata
            # Mevcut katmanın delta'sı ile bir sonraki katmanın ağırlıklarının transpozu çarpımı
            # ve bu katmanın aktivasyon fonksiyonunun türevi ile çarpımı
            delta_hidden = np.dot(delta, self.weights[f'W{i + 1}'].T) * self.relu_derivative(self.z_values[f'z{i}'])

            # Gizli katman ağırlık ve bias gradyanları
            gradients[f'dW{i}'] = np.dot(self.activations[f'a{i - 1}'].T, delta_hidden)
            gradients[f'db{i}'] = np.sum(delta_hidden, axis=0, keepdims=True)

            delta = delta_hidden  # Bir sonraki geri yayılım adımı için delta'yı güncelle

        # Ağırlık ve Bias güncellemeleri (Gradyan İniş)
        for key in self.weights:
            self.weights[key] -= self.learning_rate * gradients[f'd{key}']
            # Bias gradyanları için key'den 'W' kısmını çıkarmak gerekiyor. Örn: 'W1' -> 'b1'
            self.biases[f'b{key[1]}'] -= self.learning_rate * gradients[f'd{f"b{key[1]}"}']

    def train(self, X_train, y_train, epochs, batch_size=32):
        """
        Sinir ağı modelini eğitim verisi üzerinde eğitir.

        Args:
            X_train (np.array): Eğitim özellikleri.
            y_train (np.array): Eğitim etiketleri.
            epochs (int): Eğitim epoch sayısı.
            batch_size (int): Her eğitim adımında kullanılacak örnek sayısı.

        Returns:
            list: Her epoch için ortalama kayıp değerleri.
        """
        losses = []
        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            # Veriyi karıştır (mini-batchler için)
            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            epoch_loss = 0
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # İleri yayılım
                output = self.forward_propagation(X_batch)

                # Kayıp hesapla
                loss = self.cross_entropy_loss(y_batch, output)
                epoch_loss += loss

                # Geri yayılım ve ağırlık güncelleme (Gradyan İniş)
                self.backward_propagation(X_batch, y_batch, output)

            avg_epoch_loss = epoch_loss / (num_samples / batch_size)
            losses.append(avg_epoch_loss)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        return losses

    def predict(self, X):
        """
        Eğitilmiş model ile yeni verilere tahmin yapar.

        Args:
            X (np.array): Tahmin yapılacak giriş verisi.

        Returns:
            np.array: Her örnek için tahmin edilen sınıf etiketleri.
        """
        probabilities = self.forward_propagation(X)
        return np.argmax(probabilities, axis=1)  # En yüksek olasılığa sahip sınıfı döndür


# --- Veri Seti Hazırlığı (Iris) ---
print("--- Veri Hazırlığı ---")
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

# Veriyi eğitim ve test kümelerine ayır
# stratify=y ile her sınıftan dengeli dağılım sağlanır.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Veriyi ölçeklendir (standartlaştırma)
# Sinir ağları genellikle ölçeklendirilmiş verilerle daha iyi performans gösterir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Eğitim Verisi Boyutu (X_train_scaled): {X_train_scaled.shape}")
print(f"Test Verisi Boyutu (X_test_scaled): {X_test_scaled.shape}")
print(f"Eğitim Hedef Boyutu (y_train): {y_train.shape}")
print(f"Test Hedef Boyutu (y_test): {y_test.shape}")
print(f"Hedef Sınıf İsimleri: {target_names}")