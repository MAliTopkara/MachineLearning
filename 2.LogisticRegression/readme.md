# **Makine Öğrenmesi Laboratuvar Ödevi – Logistic Regression**
Makine Öğrenmesi laboratuvarında verilen Logistic Regression yöntemi kullanılarak ikili sınıflandırma yapılması istenen ödevi nasıl yaptığımı anlatacağım.
## **Veri Seçimi ve Hazırlık**
İlk olarak Logistic Regression yöntemi için uygun bir veri seti bulmaya çalıştım. Bu aşamada Kaggle ve UCI gibi platformlarda çeşitli veri setleri inceledim. İçlerinden örnek sayısı, özellik sayısı ve sınıflandırma için uygun etiket içermesi açısından Indian Liver Patient Dataset adlı veri setini seçtim.

Veri setinde bazı eksik veriler ve mantıksal hatalar vardı. Örneğin, "Albumin\_and\_Globulin\_Ratio" sütununda eksik değerler mevcuttu. Bu eksik veriler, sütunun ortalamasıyla doldurularak temizlendi. Ayrıca, "Gender" gibi kategorik değişkenler one-hot encoding yöntemiyle sayısal hale getirildi. Hedef değişken olan "Dataset" sütunu 1 (hasta) ve 2 (sağlıklı) değerlerinden 1 (hasta) ve 0 (sağlıklı) olarak dönüştürüldü.
## **Scikit-learn Kullanılarak Logistic Regression Modeli**
Kodda sırasıyla şu adımlar izlendi:

1\. Veri setinin yüklenmesi ve eksik değerlerin işlenmesi  
2\. Kategorik değişkenlerin sayısallaştırılması  
3\. Eğitim ve test veri setlerine ayrılması (%80 eğitim – %20 test)  
4\. Scikit-learn kullanılarak LogisticRegression modelinin oluşturulması  
5\. Modelin eğitilmesi  
6\. Test verileriyle tahmin yapılması  
7\. Modelin doğruluk, sınıflandırma raporu ve karışıklık matrisi gibi metriklerle değerlendirilmesi  
8\. Karışıklık matrisinin görselleştirilmesi
## **Scikit-learn Kullanmadan Logistic Regression**
Aynı işlemler bu kez Scikit-learn kütüphanesi kullanılmadan, NumPy ile sıfırdan Logistic Regression algoritması yazılarak uygulandı. Kodda izlenen adımlar şunlardı:

1\. Veri temizliği ve ön işleme (yukarıdaki ile aynı)  
2\. Eğitim/test ayrımı yapılması  
3\. Sigmoid fonksiyonu ve binary cross-entropy kayıp fonksiyonunun tanımlanması  
4\. Gradient descent yöntemi ile ağırlık ve bias değerlerinin güncellenmesi  
5\. Modelin test verisi ile tahmin yapması  
6\. Doğruluk oranının hesaplanması  
7\. Kayıp fonksiyonunun epoch bazlı çizilmesi  
8\. Karışıklık matrisinin oluşturulması ve görselleştirilmesi
## **Karşılaştırmalar**
### **Eğitim Verisi Ayrımı**
Veri seti, her iki model için de %80 eğitim - %20 test olacak şekilde ayrıldı. Bu oran, %70-30 veya %75-25 gibi oranlara göre doğruluk değerini daha optimum seviyeye getirdi.
### **Eğitim Süreleri**

|Model Türü|Eğitim Süresi (saniye)|
| :- | :- |
|Scikit-learn Logistic Regression|0\.003252|
|Kendi Logistic Regression Modeli|0\.001448|

Kendi yazdığım model biraz daha hızlı eğitildi. Bunun sebebi, Scikit-learn modelinin arka planda bazı ek kontroller (örneğin: regularization, solver seçimi gibi) yapması olabilir. Ancak her iki model de oldukça hızlı eğitildiği için bu fark pratikte çok büyük önem taşımamaktadır.
### **Test Süreleri**

|Model Türü|Test Süresi (saniye)|
| :- | :- |
|Scikit-learn Logistic Regression|0\.000900|
|Kendi Logistic Regression Modeli|0\.001771|

Test aşamasında Scikit-learn modeli daha hızlı çalıştı. Kendi modelimiz saf NumPy işlemleriyle çalıştığı ve daha az optimizasyon içerdiği için biraz daha yavaş olabilir. Yine de test süreleri milisaniyeler düzeyindedir.
### **Doğruluk Karşılaştırması**

|Model Türü|Doğruluk Oranı|
| :- | :- |
|Scikit-learn Logistic Regression|%76.55|
|Kendi Logistic Regression Modeli|%76.55|

Her iki model de aynı doğruluk oranını vermektedir. Bu, kendi yazdığım Logistic Regression algoritmasının doğru bir şekilde çalıştığını ve beklenen sonuçları ürettiğini göstermektedir.
## **Sonuç**
Bu ödevde hem Scikit-learn ile hem de kütüphane kullanmadan Logistic Regression algoritması uygulanarak başarılı bir ikili sınıflandırma gerçekleştirilmiştir. Her iki yöntem de benzer doğruluk değerleri üretmiş, ancak eğitim ve test süreleri açısından farklılıklar gözlemlenmiştir. Bu karşılaştırmalar, algoritma öğrenimi açısından hem teorik hem pratik anlamda önemli kazanımlar sağlamıştır.
