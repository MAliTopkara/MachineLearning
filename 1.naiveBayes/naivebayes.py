import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


veriler = pd.read_csv("C:\\Users\\Monster\\Downloads\\diabetes_dataset.csv")

veriler.fillna(veriler.mean(), inplace=True)

veriler = veriler.rename(columns={'Outcome': 'Sonuc'})

ozellikler = veriler.drop('Sonuc', axis=1).values
sonuc = veriler['Sonuc'].values

egitim_ozellik, test_ozellik, egitim_sonuc, test_sonuc = train_test_split(
    ozellikler, sonuc, test_size=0.2, random_state=42)


class BenimGaussianNB:
    def egit(self, ozellikler, sonuc):
        # Sınıfları belirleme
        self.siniflar = np.unique(sonuc)
        n_ozellik = ozellikler.shape[1]
        self.ortalama = {}
        self.varyans = {}
        self.oncelik = {}

        for s in self.siniflar:
            ozellikler_s = ozellikler[sonuc == s]
            self.ortalama[s] = np.array([np.sum(ozellikler_s[:, j]) / ozellikler_s.shape[0] for j in range(n_ozellik)])
            self.varyans[s] = np.array(
                [np.sum((ozellikler_s[:, j] - self.ortalama[s][j]) ** 2) / ozellikler_s.shape[0] for j in
                 range(n_ozellik)])
            self.oncelik[s] = ozellikler_s.shape[0] / float(ozellikler.shape[0])

    def _gaus_olasilik(self, sinif, ornek):
        ort = self.ortalama[sinif]
        vary = self.varyans[sinif]
        pay = np.exp(- ((ornek - ort) ** 2) / (2 * vary + 1e-9))
        payda = np.sqrt(2 * np.pi * vary + 1e-9)
        return pay / payda

    def tahmin_et(self, ozellikler):
        return np.array([self._tek_tahmin(ornek) for ornek in ozellikler])

    def _tek_tahmin(self, ornek):
        posteriors = []
        for s in self.siniflar:
            log_oncelik = np.log(self.oncelik[s])
            log_olasilik = np.sum(np.log(self._gaus_olasilik(s, ornek) + 1e-9))
            posteriors.append(log_oncelik + log_olasilik)
        return self.siniflar[np.argmax(posteriors)]

print("*** Manuel Gaussian Naive Bayes Modelimiz ***")
manuel_gnb = BenimGaussianNB()

baslangic = time.time()
manuel_gnb.egit(egitim_ozellik, egitim_sonuc)
egitim_suresi = time.time() - baslangic

baslangic = time.time()
tahmin_sonuc = manuel_gnb.tahmin_et(test_ozellik)
test_suresi = time.time() - baslangic

print("Egitim suresi: {:.6f} saniye".format(egitim_suresi))
print("Test suresi: {:.6f} saniye".format(test_suresi))
dogru = 0
karmasik_matris = [[0, 0], [0, 0]]  # [ [Gerçek 0 tahmin 0, Gerçek 0 tahmin 1], [Gerçek 1 tahmin 0, Gerçek 1 tahmin 1] ]

for gercek, tahmin in zip(test_sonuc, tahmin_sonuc):
    if gercek == tahmin:
        dogru += 1
    if gercek == 0 and tahmin == 0:
        karmasik_matris[0][0] += 1
    elif gercek == 0 and tahmin == 1:
        karmasik_matris[0][1] += 1
    elif gercek == 1 and tahmin == 0:
        karmasik_matris[1][0] += 1
    elif gercek == 1 and tahmin == 1:
        karmasik_matris[1][1] += 1

dogruluk = dogru / len(test_sonuc)
print("Dogruluk: {:.2f}".format(dogruluk))
print("Karmasiklik Matrisi :")
print(np.array(karmasik_matris))
tik_pozisyon = np.arange(2)
plt.figure(figsize=(6, 5))
plt.imshow(karmasik_matris, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Benim GaussianNB - Karmaşıklık Matrisi")
plt.colorbar()
plt.xticks(tik_pozisyon, ['0', '1'])
plt.yticks(tik_pozisyon, ['0', '1'])
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
for i in range(2):
    for j in range(2):
        renk = "white" if karmasik_matris[i][j] > np.max(karmasik_matris) / 2. else "black"
        plt.text(j, i, karmasik_matris[i][j], horizontalalignment="center", color=renk)
plt.show()