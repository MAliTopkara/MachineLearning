3.Laboratuvar Ödevi

Tek readme dosyası kullanarak soruları açıklamayı düşündüm.


 # **Soru 1 - Matris Manipülasyonu, Özdeğerler ve Özvektörlerin Makine Öğrenmesindeki Rolü**
## **Matris Manipülasyonu Nedir?**
Matris manipülasyonu, verilerin matris (satır–sütun) biçiminde işlenmesiyle yapılan temel işlemlerdir: toplama, çarpma, transpoz alma, tersini bulma, dilimleme ve yeniden şekillendirme gibi işlemleri kapsar.
## **Özdeğer ve Özvektör Nedir?**
Bir kare matris A için A \* v = λ \* v denkleminde λ özdeğer, v ise o özdeğere ait özvektördür. Bu vektör, matrisle çarpıldığında yönü değişmeden sadece ölçeklenen bir vektördür.
## **Makine Öğrenmesinde Nasıl Kullanılır?**
Matris manipülasyonu ve özdeğer/özvektör analizleri birçok makine öğrenmesi yönteminde kullanılmaktadır:
## **YÖNTEMLER:**
## **PCA (Principal Component Analysis)**
Veri kümesinin kovaryans matrisi üzerinden yapılan özdeğer analizleri ile boyut indirgeme sağlanır. En büyük özdeğere karşılık gelen yön, en fazla varyansı taşır.
## **SVD (Singular Value Decomposition)**
Veri matrisinin ayrıştırılması ile düşük boyutlu temsiller elde edilir. Görüntü sıkıştırma ve öneri sistemlerinde kullanılır.
## **Spektral Kümeleme**
Veri noktalarının benzerlik grafiğinden elde edilen Laplasyen matrisinin özvektörleri kullanılarak veriler alt uzayda gömülür ve kümelenir.
## **LDA (Linear Discriminant Analysis)**
Sınıflar arası ayrımı en iyi yapan yönlerin bulunmasında, sınıf içi ve sınıflar arası scatter matrislerinin oranının özdeğer çözümlemesi yapılır.
## **Graf Merkeziliği (PageRank, Eigenvector Centrality)**
Bir ağdaki düğümlerin önemini belirlemek için komşuluk matrisinin özvektörleri analiz edilir.
## **Kaynaklar**
Jason Brownlee - Introduction to Matrices for Machine Learning <https://machinelearningmastery.com/introduction-matrices-machine-learning/>
# **Soru 2 - NumPy’ın `linalg.eig` Fonksiyonu: Dokümantasyon ve Kaynak Kod İncelemesi**
## **Fonksiyon Tanımı**
numpy.linalg.eig(a) fonksiyonu, kare bir matrisin özdeğerlerini ve bu özdeğerlere karşılık gelen özvektörleri hesaplar.

Girdi:
\- a : (n x n) boyutlu kare matris veya çoklu matris içeren diziler (broadcast destekli)

Çıktı:
\- w : Karmaşık özdeğerler dizisi
\- v : Sütunları özvektörler olan matris
(Yani A @ v[:, i] ≈ w[i] \* v[:, i])

Uyarılar:
\- Simetrik/Hermityan matrislerde eig yerine eigh kullanılması önerilir.
\- Karmaşık olmayan matrisler için bile sonuç karmaşık dtype ile dönebilir.
## **İç Yapısı ve Akışı**
1\. Python seviyesi (numpy/linalg/\_linalg.py):

- Kare matris kontrolü

- Tip dönüşümleri

- Gufunc çağrısı


2\. C seviyesi:

- NumPy'nin LAPACK wrapper'ı üzerinden LAPACK fonksiyonu çağrılır:

-Gerçek matris → DGEEV

-Karmaşık matris → ZGEEV


3\. LAPACK (Fortran):

- Hessenberg formuna indirme

- QR iterasyonu ile özdeğer çıkarma

- Geri çözümleme ile özvektör üretimi

## **Örnek Kullanım**
import numpy as np

A = np.array([[4, 2, 2],

             [0, 3, -2],
             
             [0, 1, 1]], dtype=float)

w, v = np.linalg.eig(A)

\# w: özdeğerler, v: özvektörler
\# Doğrulama: A @ v[:, i] ≈ w[i] \* v[:, i]
## **Kaynaklar**
<https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html>

<https://github.com/numpy/numpy/blob/main/numpy/linalg/_linalg.py>



# **Soru 3 - Özdeğer Hesaplamalarının Manuel ve NumPy Karşılaştırması**
Bu soruda, 3x3 boyutlu bir matris için hem manuel yöntemle hem de NumPy kütüphanesinin `linalg.eig` fonksiyonu kullanılarak özdeğer ve özvektör hesaplaması yapılmış, sonuçlar karşılaştırılmıştır.
Kullanılan matris:

A = [[6, 1, -1],

    [0, 7, 0],

    [3, -1, 2]]
## **Manuel Hesaplama Sonuçları**
Özdeğerler: [7. 5. 3.]

Özvektörler:
[[ 0.5883,  0.7071,  0.3162],

[ 0.7845,  0.0000, -0.0000],

[ 0.1961,  0.7071,  0.9487]]

Hesaplama süresi: 0.000518 saniye
## **NumPy ile Hesaplama Sonuçları**
Özdeğerler: [5. 3. 7.]

Özvektörler:
[[0.7071, 0.3162, 0.5883],

[0.0000, 0.0000, 0.7845],

[0.7071, 0.9487, 0.1961]]

Hesaplama süresi: 0.000171 saniye
## **Karşılaştırma**

|Özellik|Sonuç|
| :- | :- |
|Özdeğerler yaklaşık eşit mi?|Evet (True)|
|Özvektörler yaklaşık eşit mi?|Hayır (False)|
|Hesaplama Süresi|NumPy daha hızlı|
## **Yorum**
Manuel yöntem ile NumPy fonksiyonunun özdeğer sonuçları tamamen uyumludur. Özvektörlerde ise yön ve normalize farklarından dolayı yaklaşık eşleşme sağlanamamıştır. NumPy daha hızlı sonuç vermektedir. Büyük boyutlu matrislerde bu hız farkı daha da önemli hale gelir.
