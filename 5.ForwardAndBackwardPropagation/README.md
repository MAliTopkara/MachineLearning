# Forward and Backward Propagation with Neural Network (Iris Dataset)

Bu proje, YZM212 Makine Öğrenmesi dersi kapsamında verilen 6. Laboratuvar ödevi için hazırlanmıştır. Projede, sinir ağı (neural network) mimarisi kullanılarak ileri (forward) ve geri (backward) yayılım algoritmaları sıfırdan (scratch) kodlanmış ve Iris veri seti üzerinde test edilmiştir.

##  Projenin Amacı

Sinir ağı yapı taşlarını hazır kütüphaneler kullanmadan manuel olarak geliştirerek;  
- ileri yayılım algoritması,
- geri yayılım (backpropagation),
- aktivasyon fonksiyonları,
- gradyan iniş (gradient descent)  
mekanizmalarının nasıl çalıştığını uygulamalı olarak öğrenmek amaçlanmıştır.

## Kullanılan Veri Seti: Iris Dataset

- **Veri Seti Kaynağı:** scikit-learn (`sklearn.datasets.load_iris()`)
- **Amaç:** Çiçeklerin yaprak ve taç yaprağı uzunluk/genişlik ölçümlerine göre türünü (Setosa, Versicolor, Virginica) sınıflandırmak.
- **Özellik Sayısı:** 4
- **Sınıf Sayısı:** 3
- **Neden tercih edildi?**
  - Dengeli ve küçük bir veri setidir.
  - 3 sınıflı (multi-class) olması, softmax + çapraz entropi kombinasyonu için uygundur.
  - Görselleştirme ve eğitim süresi açısından hızlı ve etkilidir.
  - Sinir ağı eğitimi için veri ön işleme (ölçeklendirme) kolaydır.

##  Sinir Ağı Mimarisi

Model `NeuralNetwork` sınıfı olarak yazılmıştır. Hiçbir hazır makine öğrenmesi modeli kullanılmamıştır.

- **Giriş Katmanı:** 4 nöron
- **Gizli Katman 1:** 10 nöron (ReLU aktivasyon)
- **Gizli Katman 2:** 5 nöron (ReLU aktivasyon)
- **Çıkış Katmanı:** 3 nöron (Softmax aktivasyon)
- **Kayıp Fonksiyonu:** Çapraz Entropi (Cross-Entropy)
- **Öğrenme Oranı:** 0.01
- **Epoch Sayısı:** 1000
- **Batch Size:** 32

## Kullanılan Kütüphaneler

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn` (sadece veri seti yükleme, preprocessing ve metric hesaplamaları için)

##  Sonuçlar

- **Eğitim Doğruluğu:** ~%100
- **Test Doğruluğu:** ~%96.67

###  Karmaşıklık Matrisi (Test Seti)
https://github.com/MAliTopkara/MachineLearning/blob/main/5.ForwardAndBackwardPropagation/labg%C3%B6rsel2.png?raw=true

### Kayıp Eğrisi (Loss vs Epoch)
https://github.com/MAliTopkara/MachineLearning/blob/main/5.ForwardAndBackwardPropagation/labg%C3%B6rsel1.png?raw=true

##  Eğitim Süreci

1. Veriler standartlaştırıldı (`StandardScaler`)
2. Sinir ağı yapısı kuruldu
3. İleri ve geri yayılım döngüsü ile model eğitildi
4. Model değerlendirmesi yapıldı (accuracy, loss grafiği, confusion matrix)
5. Görselleştirmeler oluşturuldu
