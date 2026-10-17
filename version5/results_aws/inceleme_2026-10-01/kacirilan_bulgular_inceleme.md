# TMLR taslağı: ham sonuçlarda kaçırılan bulgular

**İncelenen taslak:** `results_aws/tmlr_draft/main.tex` (30 Eylül 2026)  
**Veri:** `results_aws/` içindeki 32 CSV (13.620 koşu) ve sweep logları. Yeni eğitim koşusu yapılmadı.  
**Yeniden üretim:** `python3 missed_findings_check.py` → çıktısı `missed_findings_output.txt`. Kalibrasyon ortalamaları `sweep.log` / `stage2.log` satırlarıyla birebir eşleşiyor.

Tohum sayıları: lrsweep/main 30 tohum, kalibrasyon 6 tohum (101–106). Kalibrasyona dayanan maddeler keşif niteliğindedir.

---

## Özet

| # | Bulgu | Taslağa etkisi | Önem |
|---|---|---|---|
| 1 | Kalibrasyon ızgarası, stabilite sonucunu 6 genişliğe taşıyor. Ama H=512'de hibrit de büyük kayıp yaşıyor | Ana iddiayı hem genişletir hem sınırlar | Yüksek |
| 2 | Çöken ReLU koşularında kayıp ln 10'a oturuyor, yani çıktı girdiden bağımsız hale geliyor | "Mekanizma hakkında bir şey söyleyemiyoruz" kısmını değiştirir | Yüksek |
| 3 | İki farklı çöküş türü var: ReLU'da yavaş ölüm, leaky-ReLU'da ani kayıp sıçraması | Leaky-ReLU kontrolünün yorumunu zenginleştirir | Yüksek |
| 4 | "Başarısızlık yok" ölçütü kısmi çöküşü gizliyor; 2,5× oranı 0,05'teki 23 pp'lik kayba dayanıyor | Soyut ve 4.2'deki iddia düzeltilmeli | Yüksek |
| 5 | Hibritin LR eğrisi daha çok sağa kayıyor, belirgin şekilde genişlemiyor | Stabilite yorumuna önemli bir uyarı ekler | Orta |
| 6 | MSE altında ReLU LR=0,03'te çökmüyor | Çöküşün kayıp/çıktı çiftine bağlı olduğunu gösterir | Orta |
| 7 | CIFAR'da ReLU geç değil, epoch 1'de çöküyor; LR=0,01'de bozulmayı hibrit kendisi getiriyor | CIFAR karşılaştırmasının çerçevesini değiştirir | Orta |
| 8 | Ortak başlangıç ağırlıkları ve veri sırası **zaten** kullanılıyor | "Sınırlılıklar" bölümündeki öneri yanlış; eşleştirilmiş tasarım iddia edilenden güçlü | Orta |
| 9 | Son epoch ile checkpoint farkı: CIFAR'da hibrit eğitimi aynı LR'de daha iyi bitiriyor | İsteğe bağlı ek analiz | Düşük–orta |
| 10 | Küçük düzeltmeler ve olumsuz sonuçlar | Cümle düzeyinde | Düşük |

---

## 1. Kalibrasyon ızgarası stabilite sonucunu genişletiyor ve sınırlarını gösteriyor

Taslak stabilite iddiasını yalnızca H=64/128 lrsweep verisine dayandırıyor. Kalibrasyon verisi (6 genişlik × 8 LR × 7 model × 6 tohum) Ek'te ısı haritası olarak gösteriliyor, ama metinde hiç analiz edilmiyor.

**MNIST, ortalama son doğrulama doğruluğu (%) ve son skoru <15 olan koşu sayısı (/6):**

| LR | Model | H=12 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| 0,03 | ReLU | 50,8 (0) | 89,6 (0) | 51,6 (3) | 10,4 (6) | 10,7 (6) | 10,3 (6) |
| 0,03 | tanh+ReLU | 89,3 (0) | 95,1 (0) | 96,3 (0) | 96,8 (0) | 94,8 (0) | **72,8 (0)** |
| 0,03 | tanh | 91,9 | 94,8 | 96,3 | 94,8 | 91,4 | 81,8 |
| 0,1 | ReLU | 11,3 (5) | 11,7 (5) | 10,5 (6) | 10,5 (6) | 10,5 (6) | 11,6 (5) |
| 0,1 | tanh+ReLU | 82,5 (0) | 85,0 (0) | 50,5 (0) | 35,4 (1) | 34,3 (1) | 36,8 (0) |

- **Destekleyici:** ReLU'nun LR=0,03'teki çöküşü H=64–512 aralığının tamamında görülüyor. Hibrit ise hiçbir genişlikte <%15'e düşmüyor. Etki yalnızca H=128'e özgü değil.
- **Sınırlayıcı:** Hibritin son skoru genişlik arttıkça düşüyor. H=512'de 6 koşunun 6'sı da epoch 2–4'te zirve yapıyor (~%95), sonra %58–79'a geriliyor. Hibrit burada "yüksek doğruluğu korumuyor"; yalnızca tam çöküşten kaçınıyor.
- **CIFAR-10, LR=0,03:** H≤128'de hibrit çöküşü belirgin biçimde azaltıyor (ReLU 4–6/6, hibrit 0–1/6). H=256 ve 512'de ise hibrit de 6/6 çöküyor. Yani CIFAR'daki koruma genişlikle kayboluyor. Saf tanh CIFAR H=512'de LR=0,01'de bile 3/6 çöküyor.
- **Olası açıklama:** Kararlı LR üst sınırının genişlikle düşmesi, standart parametrizasyonda beklenen bir davranış (çıktı katmanının güncellemesi H girişin toplamıdır). Bu, "Neden" bölümüne literatür bağlantısıyla eklenebilir.

**Öneri:** 4.1'e kısa bir "Width dependence" paragrafı ve küçük bir tablo ekleyin. Kalibrasyon verisinin 6 tohum olduğunu ve keşif amaçlı kullanıldığını açıkça yazın.

## 2. Çökme mekanizmasına dair loglarda doğrudan bir ipucu var

Taslak "gizli birim aktivitesini ölçmüyoruz, nedeni söyleyemeyiz" diyor. Oysa `epoch_loss` sütunu güçlü bir ipucu veriyor:

- MNIST H=128, LR=0,03'te çöken 30 ReLU koşusunun hepsinde son eğitim CE kaybı **2,3081 ± 0,0003**. Bu değer, sabit bir tahmin yapan modelin kaybına (ln 10 = 2,3026) neredeyse eşit. Tohumlar arası yayılım da çok küçük.
- Bu, ağın çıktısının girdiden bağımsız hale geldiğini gösteriyor. ReLU ağında bunun en yalın açıklaması şu: bir gizli katmandaki tüm ReLU birimleri, tüm girdiler için sıfır veriyor ("ölü katman"). tanh birimleri sıfıra kilitlenemediği için, karma katmanda girdiye bağlı bir yol her zaman kalıyor.
- Bu doğrudan bir ölçüm değil; ama olası mekanizmaları ciddi biçimde daraltıyor ve Lu et al. (2019) tartışmasını somut hale getiriyor.

**Öneri (EN):** *"In every collapsed ReLU run, the final training loss is 2.308 ± 0.0003, essentially the loss of a constant predictor (ln 10 ≈ 2.303). The network output has therefore become input-independent, which is consistent with all rectified units in a hidden layer becoming inactive. Mixed layers retain tanh units that cannot become exactly zero."*

## 3. İki farklı çöküş türü

| | ReLU (yavaş ölüm) | Leaky-ReLU ve tanh+leaky-ReLU (ani sıçrama) |
|---|---|---|
| Kayıp seyri | Minimum epoch 6–13'te (medyan 9); ardından tedricen 2,31'e yükseliyor | Tek bir epochta kayıp +3 ile +6 arası sıçrıyor |
| Zamanlama (H=128) | val<%15 ilk kez epoch 17–28'de | **30/30 koşuda** epoch 13–15'te (T+L: 23–26) |
| Son durum | Sabit tahmin, ~%10 | ~%85'te plato, LR 0,02–0,3 arasında aynı |
| Son CE kaybı | 2,31 | **3,6–4,3** (> ln 10) |

- **Erken uyarı:** ReLU'da eğitim kaybı, doğrulama çöküşünden ortalama **12,5 epoch önce** (aralık 7–17) yükselmeye başlıyor. Pratik açıdan anlatmaya değer bir sinyal.
- **Aşırı güven:** Leaky modeller %85 doğrulukta, sabit bir tahminciden bile daha yüksek CE kaybına sahip. Bu, yanlış tahminlerin çok yüksek güvenle yapıldığını gösterir. "Başarısızlık sayısı ile kayıp ikisi de gerekli" argümanını güçlendiriyor.
- **Eşzamanlılık:** Her tohumun başlangıç ağırlıkları ve veri sırası farklı. Buna rağmen sıçrama tüm tohumlarda 3 epochluk bir pencereye düşüyor. Bu, şanssız bir örnekten çok, deterministik bir sürüklenmeye (ör. ağırlık normunun büyümesi) işaret ediyor. Gelecek çalışma için iyi bir hipotez.
- Tanh+ReLU koşularında kayıp epoch 40'ta hâlâ düşüyor (minimum epoch medyanı 38). Yani hibritteki stabilite "ertelenmiş bir çöküş" değil. Bu destekleyici bir kontroldür.

## 4. "Başarısızlık yok" penceresi kısmi çöküşü gizliyor

Soyuttaki ve 4.2'deki "üst oran 0,02'den 0,05'e çıkıyor (2,5×)" ifadesi, LR=0,05'i hibrit için geçerli sayıyor. Ama MNIST H=128, LR=0,05'te:

- Hibritin **30/30** koşusu zirveden >10 pp kaybediyor. Ortalama son skor %69,3, kayıp 23,2 pp; en iyi epoch ortalaması 2,7.
- Aynı LR'de saf tanh %86,4 (1/30), leaky-ReLU %86,4. Hibrit, tanh ebeveyninden açıkça kötü.

**Peak→final kaybı >10 pp olan koşu sayısı (/30, MNIST):**

| H | Model | 0,02 | 0,03 | 0,05 | 0,1 |
|---|---|---|---|---|---|
| 64 | ReLU | 0 | 12 | 30 | 11 |
| 64 | tanh+ReLU | 0 | 0 | 3 | 12 |
| 64 | tanh | 0 | 0 | 0 | 11 |
| 128 | ReLU | 0 | 30 | 30 | 20 |
| 128 | tanh+ReLU | 0 | 0 | **30** | 18 |
| 128 | tanh | 0 | 0 | 1 | 14 |
| 128 | leaky-ReLU | **24** | 19 | 1 | 0 |

- ">10 pp kayıp yok" ölçütüyle üst sınırlar ReLU için 0,02, hibrit için 0,03 (her iki genişlikte). Oran 2,5× değil, **1,5×**.
- Hibrit her iki ebeveynini de yalnızca LR=0,03 civarındaki dar bir bantta geçiyor. LR≥0,05'te tanh daha kararlı.
- Leaky-ReLU, H=128'de 0,02'de bile çöküyor; ReLU ise çökmüyor. Bu ilginç ve taslakta yok.

**Öneri:** windows tablosuna üçüncü bir kural ekleyin ("hiçbir koşu >10 pp kaybetmiyor"). 2,5× ifadesini soyuttan çıkarın ya da bu ölçütle birlikte verin.

## 5. Hibritin LR eğrisi sağa kayıyor; belirgin şekilde genişlemiyor

| Veri | H | Model | En iyi LR | 5 pp içinde kalan son LR | Oran |
|---|---|---|---|---|---|
| MNIST | 64 | ReLU | 0,005 | 0,02 | 4× |
| MNIST | 64 | tanh+ReLU | 0,01 | 0,03 | 3× |
| MNIST | 128 | ReLU | 0,005 | 0,02 | 4× |
| MNIST | 128 | tanh+ReLU | 0,01 | 0,03 | 3× |
| MNIST | 128 | sigmoid | 0,03 | 0,3 | 10× |

- LR=0,03, ReLU için kendi optimumunun 6 katı, hibrit için 3 katı. Sabit nominal LR'deki karşılaştırma, modelleri eğrilerinin farklı noktalarında karşılaştırıyor.
- Sigmoid'in yüksek LR dayanıklılığı, düşük LR'deki yavaşlığıyla birlikte görülüyor (LR=0,001'de epoch 1 doğruluğu ~%26, diğerlerinde ~%91). Bu, σ′ ≤ 0,25'in etkin adım boyunu küçültmesiyle tutarlı. Hibrit için de benzer bir "etkin LR" etkisi düşünülebilir.

**Öneri:** Bir şekil ya da paragraf ekleyin: x ekseni = LR / modelin kendi en iyi LR'si. Böylece stabilite kazancının ne kadarının kaymadan, ne kadarının genişlemeden geldiği görünür. "Neden" bölümüne de bir cümle eklenebilir.

## 6. MSE altında çöküş eşiği farklı

Kalibrasyon MSE verisi (6 tohum, MNIST):

| LR | Model | H=64 | H=128 |
|---|---|---|---|
| 0,03 | ReLU | 93,0 | **97,9** |
| 0,03 | tanh+ReLU | 97,7 | 98,3 |
| 0,1 | ReLU | 9,3 | 10,6 |
| 0,1 | tanh+ReLU | 75,6 | 36,3 |
| 0,1 | tanh | 93,9 | 92,7 |

- ReLU, CE ile LR=0,03'te çöküyor (%10,4), MSE+sigmoid çıktıyla çökmüyor (%97,9). Yani çöküş eşiği kayıp/çıktı çiftine bağlı.
- MSE'de hibrit tanh'tan açıkça daha az kararlı.
- Taslağın MSE bölümü yalnızca ayarlanmış doğruluğu ele alıyor; bu stabilite bulgusu eksik. Eklenirken 10× türev ölçeği notu mutlaka hatırlatılmalı.

## 7. CIFAR-10: çöküşün türü farklı ve hibrit kendi bozulmasını yaratıyor

- **LR=0,03, ReLU:** 30/30 koşu **epoch 1'den itibaren** <%15 (zirve ortalaması %15,5). Ağ hiç öğrenmiyor; bu "geç eğitim çöküşü" değil. Taslaktaki 26/30 – 9/30 karşılaştırması, MNIST'teki fenomenden farklı bir şeyi ölçüyor.
- **LR=0,03, hibrit:** önce ~%26'ya kadar öğreniyor, sonra 18/30 koşu en az bir kez <%15'e iniyor. Son epochta yalnızca 9 koşu <%15. Yani "son skor" ölçütü hibritin başarısızlıklarını eksik sayıyor.
- **LR=0,01:** ReLU ve tanh epoch 40'a kadar iyileşmeye devam ediyor (en iyi epoch medyanı 34–35). Hibrit ise epoch ~12'de zirve yapıp %34,6'dan %28,1'e geriliyor. Kalibrasyon verisinde hibrit, H=32–256 aralığında LR=0,01'de **her iki ebeveyninden de kötü**. Taslak yalnızca H=128'i söylüyor. Bu, CIFAR'da karışımın bozulmayı kendisinin getirdiği bir "negatif sinerji".

## 8. Ortak başlangıç ağırlıkları ve veri sırası zaten var

`net.cpp` içindeki `make_mask` fonksiyonu, `interleave` ve `block` düzenlerinde RNG kullanmıyor. Bu nedenle aynı tohumla çalışan tüm saf modeller ve %50 interleave hibritler (ve farklı LR'ler):

- birebir **aynı başlangıç ağırlıklarını**,
- her epochta **aynı veri sırasını** kullanıyor.

Ampirik doğrulama: CIFAR H=512'de son doğrulama artıklarının modeller arası korelasyonu ortalama **r = 0,68**. Örneğin tohum 8, yedi modelin hepsinde ortalamadan 2,0–3,6 pp düşük.

Sonuçlar:

- Taslağın satır 257'deki önerisi ("common starting weights and training orders" ile yeni deney) **zaten karşılanmış durumda**. Bunu bir sınırlılık değil, bir güçlü yön olarak yazın: *"ReLU and tanh–ReLU runs with the same seed share identical initial weights and per-epoch data orders; 30/30 versus 0/30 is therefore a controlled contrast."* (Yalnızca `random` düzen bu eşleşmeyi bozuyor; taslak bunu doğru belirtiyor.)
- Hangi ReLU tohumlarının H=64'te çöktüğü, aynı tohumların başka LR'lerdeki davranışıyla ilişkili değil (ör. 0,02'de kayıp: 1,43 – 1,44, p=0,31). Yani çöküş yalnızca başlangıç koşullarıyla belirlenmiyor.

## 9. Son epoch ile checkpoint farkı

- MNIST'te H≥64 için checkpoint seçimi test doğruluğunu artırmıyor (ortalama −0,03 ile 0 pp). CIFAR'da +0,2 ile +1,0 pp artırıyor.
- **CIFAR, son epoch testi, tanh+ReLU − ReLU (eşleştirilmiş, 95% bootstrap):** H=64 **+0,77** [0,45, 1,10]; H=128 **+0,23** [0,04, 0,42]; H=256 +0,19 [0,02, 0,38]. Checkpoint farkları daha küçük; H=128 ve 256'da aralık sıfırı içeriyor. H=128/256'da iki model de aynı LR'yi (0,0003) kullanıyor. Holm düzeltmesi uygulanmadı; keşif amaçlı.
- **MNIST H=128, LR=0,03 checkpoint testi:** ReLU %96,07, hibrit %96,77 → **+0,70 pp, 30/30 galibiyet**. Taslak "86 puanlık bir kazanç değildir" diyor ama gerçek değeri vermiyor. Bu rakam o cümleye eklenebilir.

## 10. Küçük düzeltmeler ve olumsuz sonuçlar

- **Düzeltme (4.1, onset paragrafı):** "every one of those ReLU runs predicts one class for all test images" ifadesi tam doğru değil. 26/30 koşu tamamen tek sınıf tahmin ediyor; 4 koşu 10.000 görüntünün 9.997–9.999'unda tek sınıf tahmin ediyor. Öneri: *"predicts a single class for at least 99.97% of test images (exactly one class in 26 of 30 runs)"*. Çöken sınıf da tohumdan tohuma değişiyor (1, 2, 9, …).
- **Olumsuz (raporlanabilir):** Hibritler daha hızlı yakınsamıyor. Aynı LR'de epoch 1 doğruluğu ve %97'ye ulaşma epoch'u ReLU ile aynı. `main.cpp` içindeki "the hybrids converge faster than their parents" yorumu veriyle desteklenmiyor; bu yorum makaleye taşınmamalı.
- **Sınıf düzeyi:** CIFAR'da hibrit farkı sınıflar arasında ±2 pp'lik bir yeniden dağılım şeklinde (H=64: kedi +2,0, kurbağa +2,1, gemi +2,5, araba −1,1). Tek tip bir kazanç yok. MNIST H=128'deki kazanç ise çoğu sınıfa yayılmış.
- **Süre:** tanh MNIST'te ~%5–10 ek süre getiriyor. sigmoid+ReLU'nun CIFAR H=512 süresi ReLU'nun 0,43 katı, bu da süre ölçümünün iş yüküne bağlı olduğunu gösteriyor. Taslaktaki uyarıyı destekliyor.

---

## Önerilen öncelik sırası

1. Madde 4 (soyut ve 4.2'deki 2,5× ifadesi) ve madde 10'daki onset cümlesi: doğruluk düzeltmeleri.
2. Madde 2 ve 3: mevcut veriden ek koşu gerektirmeden mekanizma tartışması.
3. Madde 1 ve 7: genişlik bağımlılığı ve CIFAR çerçevesi.
4. Madde 8: sınırlılıklar bölümündeki yanlış öneriyi güçlü bir yöne çevirmek.
5. Madde 5, 6, 9: isteğe bağlı ek analizler.
