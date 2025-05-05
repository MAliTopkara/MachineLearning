Laboratuvar 4:Lineer Regression
## Linear Regression – NumPy vs Scikit-Learn

Bu proje, YZM212 Makine Öğrenmesi labı kapsamında, öğrenci performansını tahmin etmek için doğrusal regresyon modellerinin iki farklı yöntemle uygulanmasını içermektedir. Amaç, hem elle (NumPy ile) yazılmış bir model hem de hazır scikit-learn modeli ile tahmin yaparak sonuçları karşılaştırmaktır.



## Kullanılan Veri Seti

- **Dosya**: `Student_Performance.csv`
- **Kayıt Sayısı**: 10.000 öğrenci
- **Hedef Değişken**: `Performance Index`
- **Girdi Değişkenleri**:
  - `Hours Studied`
  - `Previous Scores`
  - `Extracurricular Activities` (Yes/No → 1/0)
  - `Sleep Hours`
  - `Sample Question Papers Practiced`



## AŞAMA AŞAMA UYGULAMA

### AŞAMA 1: Veri Seçimi
- Öğrenci performansını etkileyebilecek çeşitli özellikler içeren veri seti seçildi (Multiple Linear Regression).

### AŞAMA 2: Veri İnceleme ve Ön İşleme
- `Pandas` ile veri yüklendi.
- `Extracurricular Activities` sütunu kategorik olduğundan yes ve no'lar 1 ve 0'a çevrildi.
- Giriş ve çıkış değişkenleri ayrıldı.

### AŞAMA 3: Elle Model (NumPy ile)
- En küçük kareler yöntemi kullanıldı:  
  \[
  \theta = (X^T X)^{-1} X^T y
  \]
- `NumPy` ile matris işlemleri gerçekleştirildi.
- Model katsayıları ve sabit terim hesaplandı.
- Tahminler yapıldı ve Ortalama Kare Hata (MSE) bulundu.

### AŞAMA 4: Hazır Model (scikit-learn ile)
- `LinearRegression()` sınıfı kullanılarak model eğitildi.
- Aynı veri seti ile tahminler yapıldı.
- MSE değeri elde edildi.

### AŞAMA 5: Sonuçların Karşılaştırılması

| Model Türü         | MSE (Mean Squared Error) | Açıklama |
|--------------------|---------------------------|----------|
| Elle (NumPy)       | ≈ 4.15                    | Formül ile yazılmış model |
| Scikit-learn Modeli| ≈ 4.15                    | Hazır `LinearRegression()` modeli |

Her iki modelin sonuçları neredeyse birebir aynıdır. Bu durum, kendi yazdığımız modelin doğruluğunu göstermektedir.
#### Cost Hesabı Nasıl Yapıldı?

Her iki model için de tahmin edilen değerlerle gerçek `Performance Index` değerleri arasındaki farkların karelerinin ortalaması alınarak **MSE (Mean Squared Error)** hesaplandı.  
Bu, lineer regresyonda en yaygın kullanılan **maliyet (cost) ölçütüdür** ve şu formülle hesaplanır:

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

- `y_i`: Gerçek değer  
- `ŷ_i`: Modelin tahmin ettiği değer  
- `n`: Veri sayısı  

Her iki model de yaklaşık **4.15**'lik bir MSE değeri vermiştir, bu da modellerin neredeyse aynı doğrulukta tahmin yaptığını gösterir.

## AŞAMA 6: Görselleştirme

Grafikler, oluşturduğumuz lineer regresyon modellerinin doğruluğunu ve hata dağılımını analiz etmek için kullanılmıştır. Her biri modelin farklı bir yönünü görselleştirir.


###  Scikit-learn KULLANILMADAN (NumPy ile)

#### Gerçek Değerler vs Tahmin Edilen Değerler
Modelin tahminlerinin ne kadar doğru olduğunu gösterir. Noktalar kırmızı çizgiye ne kadar yakınsa model o kadar başarılıdır.

![LineerRegression1](https://github.com/user-attachments/assets/2646a87c-5aa3-4a32-9702-e2c7d319fbf1)

#### Hataların Dağılımı (Residual Histogram)
Tahmin ile gerçek değer arasındaki farkların dağılımını gösterir. Simetrik bir dağılım modelin dengeli olduğunu gösterir.

![LineerRegression2](https://github.com/user-attachments/assets/63dfa753-7656-4cd7-82fd-7651d2faea4f)


###  Scikit-learn KULLANILARAK

#### Gerçek Değerler vs Tahmin Edilen Değerler
Hazır modelin tahmin performansı. Tahmin edilen değerlerin gerçek değerlere olan yakınlığı incelenmiştir.

![LineerRegression3](https://github.com/user-attachments/assets/4945ddd0-8d53-413f-a089-6477c572e3bb)

#### Hataların Dağılımı (Residual Histogram)
Hazır modelin tahmin hatalarının dağılımı. Hatalar genellikle sıfır etrafında ve simetrikse model başarılı sayılır.

![LineerRegression4](https://github.com/user-attachments/assets/2d7106da-4cec-4355-876c-c726f3d42d51)







