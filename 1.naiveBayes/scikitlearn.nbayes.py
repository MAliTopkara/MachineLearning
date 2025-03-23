import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix


veriler = pd.read_csv("C:\\Users\\Monster\\Downloads\\diabetes_dataset.csv")

veriler.fillna(veriler.mean(), inplace=True)


veriler = veriler.rename(columns={'Outcome': 'Sonuc'})

ozellikler = veriler.drop('Sonuc', axis=1).values
sonuc = veriler['Sonuc'].values


egitim_ozellik, test_ozellik, egitim_sonuc, test_sonuc = train_test_split(
    ozellikler, sonuc, test_size=0.2, random_state=42)

print("== Scikit-learn GaussianNB ==")
gnb = GaussianNB()


start_time = time.time()
gnb.fit(egitim_ozellik, egitim_sonuc)
fit_time = time.time() - start_time

start_time = time.time()
y_pred = gnb.predict(test_ozellik)
predict_time = time.time() - start_time

print("Egitim suresi: {:.6f} saniye".format(fit_time))
print("Test suresi: {:.6f} saniye".format(predict_time))
print("Siniflandirma raporu:\n", classification_report(test_sonuc, y_pred))

cm = confusion_matrix(test_sonuc, y_pred)
print("Karmasiklik Matrisi:\n", cm)

plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Scikit-learn GaussianNB - Karmaşıklık Matrisi")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['0', '1'])
plt.yticks(tick_marks, ['0', '1'])
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max()/2. else "black")
plt.show()