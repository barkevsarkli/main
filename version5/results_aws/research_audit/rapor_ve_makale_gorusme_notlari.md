# Hibrit aktivasyon deneyleri: rapor ve makale görüşme notları

Tarih: 13 Eylül 2026

Bu belge, deneysel sonuçların değerlendirilmesi, mevcut verilerle makale hazırlanması, sonraki çalışmanın kapsamı ve aktivasyon oranının adlandırılması üzerine yapılan görüşmeleri derler. Yeni deney sonuçları içermez. Sayısal bulgular mevcut CSV dosyalarının denetimine dayanır; öneriler ve mekanizma hipotezleri gözlemsel sonuçlardan ayrı belirtilmiştir.

## 1. Çalışmanın konusu ve mevcut deney kapsamı

Araştırma, farklı aktivasyon fonksiyonlarının aynı gizli katmanda farklı nöronlara atanmasının doğruluk, macro-F1 ve eğitim kararlılığı üzerindeki etkilerini inceler.

- Mimari: iki gizli katmanlı MLP, `D → H → H → 10`.
- Veri kümeleri: MNIST ve CIFAR-10.
- Gizli katman genişlikleri: 12, 32, 64, 128, 256 ve 512.
- Ana konfigürasyonlar: ReLU, leaky-ReLU, tanh, sigmoid, tanh+ReLU, tanh+leaky-ReLU ve sigmoid+ReLU.
- Ana değerlendirme: her konfigürasyon ve genişlik için 30 seed.
- Ek deneyler: öğrenme oranı taramaları, aktivasyon yerleşimi ve oranı, MSE kayıp fonksiyonu.
- Kalibrasyon seed’leri: 101–106; değerlendirme seed’leri: 1–30.
- Denetlenen kapsam: 32 CSV dosyasında 13.620 veri satırı; ana deneylerde 2.520 koşu ve 84 konfigürasyon–veri kümesi–genişlik hücresi.

CSV dosyalarındaki 31 yinelenen başlık satırı veri olarak değerlendirilmemiştir. Dosya içinde yinelenen iş kimliği bulunmamıştır. Farklı deney aşamaları arasında 690 yinelenen iş kimliği vardır; bunlar bağımsız tekrar sayılarak birleştirilmemelidir. Tüm satırlarda durum `ok` olarak kayıtlıdır. Doğrulamayla seçilen checkpoint’in doğruluk ve macro-F1 değerleri karışıklık matrisleriyle, seçilen epoch ise doğrulama geçmişleriyle tutarlıdır. Bu kontroller sayısal kayıt tutarlılığını doğrular; bütün uygulamanın hatasız olduğunu veya veri sızıntısı olmadığını tek başına kanıtlamaz.

## 2. Mevcut verilerle makale hazırlanabilir mi?

Değerlendirme: **Evet, mevcut verilerle kapsamı açıkça sınırlandırılmış bir ampirik karşılaştırma makalesi hazırlanabilir.** Yayın kabulü garanti edilemez. Hedef yayın yeri ve özgünlük iddiası, ilgili çalışmalarla yapılacak karşılaştırmaya bağlıdır.

Mevcut makalenin katkısı, hibrit aktivasyonların her durumda üstün olduğunu ileri sürmek yerine, hangi koşullarda fayda sağladığını ve hangi koşullarda sağlamadığını ortaya koymak olabilir. Yeni deney yapılmadan ilerlenebilir; ancak raporlama ve analizde belirlenen tutarsızlıkların giderilmesi gerekir. Bunun için yeterli veri olmayan karşılaştırmalar sınırlılık olarak bırakılmalıdır.

Önerilen temel iddia:

> Hibrit aktivasyonların etkisi ağ genişliğine, öğrenme oranına ve veri kümesine bağlıdır. Tanh–ReLU hibriti, incelenen bazı MNIST koşullarında eğitim sonundaki performans çöküşünü azaltırken, en iyi ayarlanmış homojen modellere karşı tutarlı bir doğruluk üstünlüğü göstermemektedir.

Önerilen araştırma sorusu:

> Heterojen gizli katman aktivasyonları, homojen ağlara kıyasla kararlı öğrenme oranı aralığını genişletiyor ve eğitim sırasında sonradan ortaya çıkan performans çöküşünü azaltıyor mu?

Çalışma başlığı için öneri:

*Hibrit Aktivasyonlu Çok Katmanlı Algılayıcılarda Öğrenme Oranı Duyarlılığı ve Eğitim Kararlılığı: Ampirik Bir İnceleme*

Bu başlık ve araştırma sorusu öneridir; kesinleşmiş yayın başlığı veya kanıtlanmış sonuç olarak yorumlanmamalıdır. İlk makale kendi araştırma sorusunu yanıtlayan bütünlüklü bir çalışma olmalıdır. Sonraki makale, yalnızca eksik kalan açıklamaları tamamlamak yerine yeni deneylerle ayrı bir bilimsel soruyu ele almalıdır.

## 3. Doğruluk açısından temel bulgular

Aksi belirtilmedikçe aşağıdaki test skorları doğrulama doğruluğuna göre seçilen checkpoint’e aittir. `±`, 30 seed arasındaki örneklem standart sapmasıdır; güven aralığı değildir. “Yüzde puan” iki yüzde değerin doğrudan farkını ifade eder.

| Veri kümesi | Konfigürasyon | Ortalama test doğruluğu | Yorum |
|---|---|---:|---|
| MNIST | tanh+ReLU, H=512 | %98.289 ± 0.087 | Ana deneylerde nominal en yüksek ortalama |
| MNIST | ReLU, H=512 | %98.285 ± 0.064 | Hibritle arasındaki fark çok küçük ve belirsiz |
| CIFAR-10 | ReLU, H=512 | %54.875 ± 0.469 | Ana deneylerde nominal en yüksek ortalama |

Bu sıralamalar özgün kalibrasyon kuralına, sabit veri bölmesine ve 40 epoch bütçesine koşulludur. Genel bir hibrit üstünlüğü veya istatistiksel olarak benzersiz bir kazanan göstermez.

### 3.1. MNIST, H=128: kalibrasyonun sonucu değiştirmesi

Mevcut özet rapor, sonuçları en iyi checkpoint sonuçları olarak tarif etmesine rağmen hesaplamalarda son epoch’un `test_acc` ve `test_f1` alanlarını kullanmaktadır. Çalıştırıcı ise hem son epoch hem de doğrulamayla seçilen checkpoint sonuçlarını saklamaktadır.

Öğrenme oranı kalibrasyonu da son doğrulama doğruluğunu (`val_acc`) kullanır. Dağıtıma alınacak model en iyi checkpoint ise, kalibrasyonda da bununla uyumlu bir ölçüt kullanılmalıdır.

| Karşılaştırma | ReLU doğruluğu | tanh+ReLU doğruluğu | Hibrit − ReLU | Eşleştirilmiş %95 bootstrap güven aralığı |
|---|---:|---:|---:|---|
| Özgün LR seçimi: ReLU 0.003, hibrit 0.01 | %97.860 | %98.038 | +0.178 yüzde puan | [+0.141, +0.214] |
| En iyi checkpoint ölçütüyle uyumlu seçim: ikisi de 0.01 | %98.009 | %98.038 | +0.029 yüzde puan | [−0.018, +0.082] |

İkinci karşılaştırmada eşleştirilmiş t testi `p=0.266`; hibrit 30 seed’in 17’sinde daha yüksek skor verir. Görünür avantajın yaklaşık %83.5’i azalır. Göreli hata azalması %8.30’dan yaklaşık %1.47’ye iner. Bu sonuç, kalan farkın güvenilir bir üstünlük olarak sunulmaması gerektiğini gösterir. Anlamlı fark bulunmaması eşdeğerliğin kanıtı değildir.

En iyi doğrulama doğruluğuna göre seçim yapıldığında 84 ana kalibrasyon seçiminin 17’si değişmektedir. Yeni seçilen öğrenme oranlarında mevcut değerlendirme verisi bulunmayan koşullar için sonuç uydurulmamalıdır.

### 3.2. Genişliğe bağlı karşılaştırma

Aşağıdaki farklar özgün öğrenme oranlarıyla seçilmiş checkpoint sonuçlarına dayanır:

| H | MNIST: tanh+ReLU − ReLU, yüzde puan | CIFAR-10: tanh+ReLU − ReLU, yüzde puan |
|---:|---:|---:|
| 12 | −0.307 | −0.550 |
| 32 | −0.186 | −0.385 |
| 64 | +0.004 | +0.363 |
| 128 | +0.178 | +0.178 |
| 256 | −0.001 | +0.153 |
| 512 | +0.003 | −0.164 |

MNIST H=128 farkı, yukarıdaki kalibrasyon düzeltmesiyle +0.029 yüzde puana iner. Sonuçlar genişlikle birlikte monoton artan bir hibrit üstünlüğü göstermemektedir. CIFAR-10 H=64 karşılaştırmasının nominal güven aralığı pozitif olsa da, denetimdeki 120 ana checkpoint doğruluğu karşılaştırmasına uygulanan Holm düzeltmesi sonrasında `p=0.095` olur.

Sigmoid+ReLU, iki veri kümesinde de tüm ana genişliklerde ReLU’nun altında kalır. Homojen sigmoid veya tanh’ı geçmek, tek başına en güçlü homojen alternatifi geçmek anlamına gelmez.

## 4. Dayanıklılık gözleminin bilimsel karşılığı

Mevcut verilerin desteklediği dayanıklılık türü **öğrenme oranına ve eğitim sırasında performansın bozulmasına karşı kararlılıktır**. Gürültülü girdilere, dağılım değişimine, adversarial saldırılara veya nöron kaybına dayanıklılık ölçülmemiştir. “Tüm ağ her bakımdan daha dayanıklı” ifadesi mevcut kanıtın kapsamını aşar.

En belirgin örnek MNIST, H=128, LR=0.03 koşuludur:

| Ölçüm | ReLU | tanh+ReLU |
|---|---:|---:|
| Eğitim sonundaki ortalama doğrulama doğruluğu | %10.61 | %96.76 |
| Son doğruluğu %15’in altına düşen koşular | 30/30 | 0/30 |
| Her koşunun en iyi doğrulama doğruluğunun ortalaması | %96.37 | %97.23 |

ReLU’nun önce yüksek performansa ulaşıp sonra bunu kaybetmesi, **geç eğitim çöküşü** gözlemidir. Aynı öğrenme oranında hibrit ağın yüksek doğruluğu koruması, koşula bağlı bir kararlılık avantajını destekler. H=64 ve LR=0.03 için de ReLU’da 12/30 koşu %15’in altına düşerken tanh+ReLU’da bu sayı 0/30’dur.

Sıfır gözlenen başarısızlık, gerçek başarısızlık olasılığının sıfır olduğunu kanıtlamaz. Bağımsız seed varsayımıyla 0/30 için tek taraflı %95 binom üst sınırı yaklaşık %9.5’tir.

Leaky-ReLU mutlaka karşılaştırmada bulunmalıdır; o da çöküşe direnç gösterebilir. Yalnızca “çökmüyor” ölçütü yeterli değildir. %85 doğrulukta kalmakla %97 civarında kalmak ayrılmalı; yüksek performansın korunması da ölçülmelidir.

### 4.1. Önerilen kararlılık ölçütleri

- Açıkça tanımlanmış bir başarısızlık eşiğine göre çöküş sıklığı.
- En iyi doğrulama skoruyla son skor arasındaki düşüş.
- Çöküşün gerçekleştiği epoch; epoch çözünürlüğüyle gözlenebildiği belirtilmeli.
- Referans performansın belirli toleransı içinde kalan öğrenme oranları.
- Seed’ler arası performans dağılımı ve başarısız koşuların ayrı gösterimi.

Eşikler gerekçelendirilmelidir. Veri görüldükten sonra seçilen eşik ve analizler keşifsel olarak belirtilmelidir. Mevcut rapordaki %50 eşiği CIFAR-10 için uygun bir genel çöküş tanımı değildir: 1.260 kalibre edilmiş ana koşunun 561’ini çöküş saymasına rağmen bu koşuların hiçbiri %15’in altında bitmemektedir. %15 de evrensel bir standart değil, bu denetimde kullanılan şeffaf bir şansa yakın performans göstergesidir.

### 4.2. Mekanizma hipotezleri

ReLU’nun negatif bölgede sıfır türeve sahip olması, leaky-ReLU’nun bu bölgede gradyanı koruması ve tanh’ın çıktılarını sınırlaması kararlılık farkları için olası açıklamalardır. Ancak mevcut sonuçlar, çöküşün ölü ReLU’lardan veya gradyan patlamasından kaynaklandığını doğrudan kanıtlamaz.

Mekanizmayı sınamak için inaktif nöron oranları, aktivasyon dağılımları, gradyan normları, kırpılma sıklığı ve ağırlık büyüklükleri ölçülmelidir. Gradyanın preaktivasyona göre türevini kırpmak, önceki katman aktivasyonu çok büyük olduğunda ağırlık güncellemesini mutlaka sınırlamaz.

## 5. Mevcut oran deneyleri ve yeniden koşu gereksinimi

`layout_128.csv` dosyasında %25, %50 ve %75 rectifier payı deneyleri zaten bulunmaktadır. **Bu verileri mevcut makalede raporlamak için aynı koşuları yeniden çalıştırmak gerekmez.** Bu ifade, her koşulun bağımsız olarak doğrulandığı veya ek deneylerin hiçbir zaman gerekmeyeceği anlamına gelmez.

MNIST, H=128 için doğrulamayla seçilen checkpoint sonuçları:

| İkinci aktivasyonun payı | tanh+ReLU doğruluğu | tanh+leaky-ReLU doğruluğu |
|---:|---:|---:|
| %25 | %97.957 | %97.965 |
| %50 | %98.038 | %97.986 |
| %75 | %98.052 | %98.052 |

Buradaki ikinci aktivasyon sırasıyla ReLU ve leaky-ReLU’dur. %50’den %75’e geçişte tanh+ReLU farkı +0.014 yüzde puandır; %95 güven aralığı [−0.017, +0.047]. Tanh+leaky-ReLU farkı +0.066 yüzde puandır; güven aralığı [+0.025, +0.109]. Bu aralıklar yerleşim ve oran aramaları için çoklu karşılaştırma düzeltmesi içermeyen keşifsel sonuçlardır.

Oranlar ayrı ayrı öğrenme oranı kalibrasyonundan geçirilmemiştir. Bu nedenle %75’in optimum olduğu veya genel olarak %50’den üstün olduğu ileri sürülmemelidir. Mevcut makalede sınırlı bir oran analizi sunulabilir; sistematik oran–öğrenme oranı–kararlılık çalışması sonraki makaleye bırakılabilir.

## 6. Parametre adlandırması: ratio, interleave ve rectifier

### 6.1. Homojen ve hibrit katman ayrımı

Homojen katmanda bütün nöronlar aynı aktivasyonu kullanır. %50 tanh ve %50 ReLU içeren katman heterojen veya hibrit katmandır.

### 6.2. Mevcut parametrelerin anlamı

| Parametre | İşlev |
|---|---|
| `act1` | Birinci aktivasyon fonksiyonu |
| `act2` | İkinci aktivasyon fonksiyonu |
| `ratio` | İkinci aktivasyonun atanacağı nöronların hedef payı |
| `layout` | Aktivasyon atamalarının katman içindeki yerleşimi |
| `layout=interleave` | İkinci aktivasyonu katman boyunca aralıklı biçimde dağıtan yerleşim |

Örnek:

```text
act1=tanh
act2=relu
ratio=0.7
layout=interleave
```

Bu ayar yaklaşık %70 ReLU ve %30 tanh hedefler. `interleave` oranı değil, yerleşim biçimini adlandırır. Bu nedenle oran parametresine `interleave` adının verilmesi önerilmemiştir.

Nöron sayısı tam sayı olduğundan hedef oran her genişlikte tam gerçekleşmeyebilir. Örneğin H=128 için tam %70 mümkün değildir. Hedef oran yanında gerçekleşen atama sayıları ve oranı kaydedilmeli, özellikle mevcut 0.25/0.5/0.75 ızgarası dışındaki değerlerde maske oluşturucunun sonucu doğrulanmalıdır.

### 6.3. Rectifier terimi

“Rectifier” yeni bir isim değildir. ReLU’nun açılımı “Rectified Linear Unit”tır ve rectifier terimi en azından 2011 tarihli *Deep Sparse Rectifier Neural Networks* çalışmasında açıkça kullanılmaktadır. Önceki görüşmelerde “rectifier payı”, ReLU veya leaky-ReLU atanan nöronların payını topluca ifade etmek için kullanılmıştır; yeni bir parametre adı önerisi veya yenilik iddiası değildir.

## 7. Activation ratio literatürde var mı?

**“Activation ratio” ifadesinin literatürde önceki kullanımları vardır. Bu ifadeyi ilk kez bu makalenin tanıttığı iddia edilmemelidir.** Ancak önceki örneklerdeki anlam, mevcut çalışmadaki nöronlara aktivasyon fonksiyonu atama oranıyla aynı değildir.

İncelenen örnekler:

- *Identification of untrained class data using neuron clusters* (2023): “neuron activation ratio”, belirli bir sınıfın örneklerinin ne kadarının bir nöronu aktive ettiğini ifade eder. Bu, nöronun veri üzerindeki aktif olma sıklığıdır.
- *Using Summary Layers to Probe Neural Network Behaviour* (2020): “activation ratio”, iki özet nöronun aktivasyon değerlerinin oranı olarak kullanılır.

Mevcut projede ise oran, girdiye bağlı bir aktivasyon istatistiği değil, **nöronlara hangi fonksiyonun atanacağını belirleyen mimari hiperparametredir**.

| İfade | Değerlendirme |
|---|---|
| “Activation ratio terimini ilk kez biz kullanıyoruz.” | Önceki kullanımlar nedeniyle uygun değil. |
| “Bu çalışmada aktivasyon atama oranını tanımlıyor ve inceliyoruz.” | Çalışmanın değişken tanımı olarak uygun. |
| “Farklı aktivasyonlara ayrılan nöron oranını ilk kez biz araştırıyoruz.” | Yapılan sınırlı taramayla doğrulanmadı; daha kapsamlı kavramsal tarama gerekir. |

Bir ifadenin aynısının bulunamaması, altında yatan fikrin yeni olduğunu kanıtlamaz. Önceki çalışmalar aynı kavramı başka adlarla incelemiş olabilir. Özgünlük, yalnızca parametre adına veya kısaltmasına dayandırılmamalıdır.

### 7.1. Görüşmeler sonunda önerilen adlandırma

Makale için **activation allocation ratio (ρ)**, Türkçe karşılığıyla **aktivasyon atama oranı** önerilmiştir. Bu, çalışmaya açıklık kazandırmak için önerilen bir adlandırmadır; literatürdeki yerleşik standart veya ilk kez ortaya atılmış bir kavram olduğu ileri sürülmemektedir.

Kodda mevcut `ratio` adı korunabilir. Daha açıklayıcı bir arayüz istenirse `activation_ratio` veya ikinci aktivasyon payını açık eden bir isim kullanılabilir; bu görüşmeler sırasında kodda bir yeniden adlandırma yapılmamıştır. Yalnızca ReLU’ya özgü bir deney için `relu_fraction` da açık bir seçenektir.

Gerçekleşen atama oranı şu şekilde tanımlanabilir:

$$
\rho_l=\frac{N_{\mathrm{act2},l}}{N_{\mathrm{hidden},l}}.
$$

Burada `l` katman indeksidir. Hedef oranla tam sayı atamaları sonucunda gerçekleşen oran arasındaki fark ayrıca belirtilmelidir. Mevcut ana deneylerde iki gizli katmanda aynı oran kullanılmaktadır.

Makale için önerilen İngilizce tanım:

> We define the activation allocation ratio ρ as the fraction of neurons assigned to the second activation function within each hidden layer.

“We define”, çalışmada kullanılan değişkeni tanımlamayı ifade eder; tarihsel öncelik iddiası değildir.

Bu yaklaşımda her nörona ayrı bir aktivasyon atanır. Her nöronda `ρ·ReLU(x) + (1−ρ)·tanh(x)` hesaplanmaz. Dolayısıyla nöron atama oranı, tek bir nöron içindeki fonksiyon karışım katsayısından ayrılmalıdır.

## 8. Mevcut raporda giderilmesi gereken teknik noktalar

1. **Son epoch ve checkpoint sonuçları:** `test_acc`/`test_f1` ile `test_acc_best`/`test_f1_best` ayrı başlıklarla sunulmalı. Eğitim–test farkı hesaplanırken aynı checkpoint’in temiz eğitim değerlendirmesi kullanılmalı.
2. **Kalibrasyon ölçütü:** En iyi checkpoint kullanılacaksa LR seçimi de bununla uyumlu olmalı. Mevcut verilerle yapılan düzeltmeler, bağımsız yeni deney doğrulaması olarak sunulmamalı.
3. **Çöküş tanımı:** CIFAR-10’da %50 eşiğinin ürettiği yanlış yorum giderilmeli; şansa yakın başarısızlık ile performans gerilemesi ayrılmalı.
4. **Başlangıçlandırma:** İncelenen kaynak kod `U(−1/√fan_in, +1/√fan_in)` dağılımını kullanır; varyans `1/(3·fan_in)` olur. Bu, standart ReLU He varyansı `2/fan_in` değerinden altı kat küçüktür. Yöntem gerçek uygulandığı biçimiyle tarif edilmeli.
5. **MSE ölçeği:** Kaynak kod kaybı on çıktı üzerinden ortalama olarak kaydederken gradyanda ona bölmeyi atlar. Güncelleme, kaydedilen amaca göre on kat ölçeklidir. Sonuçlar bu uygulama ayrıntısı belirtilmeden standart normalize MSE karşılaştırması olarak sunulmamalı.
6. **MSE sonuçlarının kapsamı:** H=64 sonuçları da elde vardır. Hibritin ReLU’ya göre farkı H=64’te −0.202 yüzde puan, H=128’de −0.034 yüzde puandır. MSE altında genel hibrit üstünlüğü desteklenmez. H=64 hibrit LR seçimi de checkpoint ölçütüne göre değişmektedir.
7. **MNIST bölmesi:** Test bölümü 60.000 örneklik eğitim dosyasından ayrılmıştır; resmî MNIST test kümesi değildir. CIFAR-10’da resmî test bölümü kullanılır.
8. **İstatistiksel kapsam:** 30 seed, tek veri bölmesine koşullu eğitim değişkenliğini ölçer. Otuz bağımsız test kümesi değildir. Bootstrap aralıkları veri bölmesi, kalibrasyon ve uyarlamalı model seçimi belirsizliğinin tamamını kapsamaz.
9. **Çoklu karşılaştırmalar:** Ana checkpoint doğruluğu için sunulan Holm düzeltmesi 120 karşılaştırmayı kapsar. Diğer endpoint, oran, yerleşim ve LR analizleri otomatik olarak bu düzeltmenin kapsamına girmez.
10. **Yerleşim ve rastgelelik:** Rastgele maske oluşturma, ağırlık başlangıçlandırması ve eğitim sırası için kullanılan RNG akışını etkiler. Aynı seed, random yerleşim karşılaştırmalarında bu değişkenlerin aynı kaldığını garanti etmez. Tam bağlı ağlarda uygun ağırlık permütasyonlarıyla nöron yerleşimleri işlevsel olarak eşdeğer olabilir.
11. **Erken durdurma ifadesi:** Çalıştırıcı tüm 40 epoch’u tamamlar, sonra en iyi checkpoint’i değerlendirir. Bu işlem hesaplamayı erken durdurmaz; tek başına süre tasarrufu iddiasını desteklemez.
12. **Kaynak–çalıştırılabilir dosya eşleşmesi:** Uygulama bulguları yan dizindeki incelenen kaynak koda dayanır. Denetim sırasında AWS çalıştırılabilir dosyası yeniden derlenip kaynakla eşleştirilmemiştir.

## 9. Verimlilik ve konfigürasyon seçimi

Parametre sayısı, bias terimleri dâhil:

$$
P(H,D)=H^2+(D+12)H+10.
$$

Aynı genişlikte aktivasyon varyantlarının parametre sayısı aynıdır; aktivasyon hesaplama maliyetleri aynı olmak zorunda değildir.

| Veri kümesi ve konfigürasyon | Doğruluk | Parametre | Ortalama kayıtlı süre, saniye |
|---|---:|---:|---:|
| MNIST, ReLU-256 | %98.214 | 269.322 | 482,5 |
| MNIST, tanh+ReLU-512 | %98.289 | 669.706 | 1.474,4 |
| CIFAR-10, ReLU-256 | %54.074 | 855.050 | 2.240,3 |
| CIFAR-10, ReLU-512 | %54.875 | 1.841.162 | 8.400,7 |

H=256 ReLU, uygulamaya özgü maliyet hedefi verilmediğinde makul bir geçici doğruluk–kaynak dengesi seçeneğidir. MNIST karşılaştırmasında yaklaşık %59.8 daha az parametre ve %67.3 daha az kayıtlı süre, CIFAR-10 karşılaştırmasında %53.6 daha az parametre ve %73.3 daha az kayıtlı süre kullanır.

Kayıtlı süreler veri yüklemeyi, eğitimi ve değerlendirmeleri içerir; deneyler yüksek eşzamanlı iş yükü altında yürütülmüştür. Bunlar doğrudan çıkarım gecikmesi veya izole eğitim performansı değildir. Kesin verimlilik üstünlüğü için kontrollü süre, bellek ve enerji ölçümü gerekir.

## 10. İlk ve sonraki makalenin kapsam ayrımı

| Mevcut verilerle ilk makale | Yeni deneylerle sonraki makale |
|---|---|
| Hangi koşullarda kararlılık farkı var? | Farkın altında hangi mekanizma var? |
| Genişlik ve LR’ye göre karşılaştırma | Aktivasyon ve gradyan dinamiklerinin ölçümü |
| Homojen ebeveynlerle doğruluk ve performans çöküşü analizi | Başlangıçlandırma ve gradyan kırpmanın kontrollü etkileri |
| Mevcut %25/%50/%75 deneylerinin sınırlı analizi | Sistematik oran × LR × kararlılık çalışması |
| İncelenen iki gizli katmanlı MLP kapsamı | Yeni mimariler, veri bölmeleri ve dayanıklılık türleri |

Sonraki çalışma için önerilen oran odaklı soru:

> Eğitim kararlılığını korumak için katmanda ne kadar tanh bulunması gerekiyor ve bunun doğruluk maliyeti nedir?

İlk tarama için ReLU payları `0, 0.25, 0.50, 0.75, 1` olabilir; uç noktalar saf tanh ve saf ReLU kontrolleridir. İlginç bölgeler daha sonra `0.6, 0.7, 0.8` gibi değerlerle sıklaştırılabilir. Mevcut oran verileri yeniden kullanılabilir, ancak yeni çalışma için veri kümesi, genişlik, LR ve seed kapsamının eşleşmesi kontrol edilmelidir.

İki deney amacı ayrılmalıdır:

- **Aynı LR’de kararlılık:** Oranlar aynı LR’lerde karşılaştırılır; eğitim çöküşü ve performans kaybı incelenir.
- **Ayrı ayarlanmış en iyi performans:** Her oran için eşit kalibrasyon bütçesiyle LR seçilir; en iyi elde edilebilir sonuç karşılaştırılır.

İlk aşamada iki gizli katmanda aynı oranı kullanmak yorumu kolaylaştırır. Katman başına farklı oranlar sonraki bir ablation olabilir. Aktivasyon atama oranı adını değiştirmek veya mevcut sonuçları yeniden sunmak için yeni koşu gerekmez.

## 11. Önerilen çalışma sırası

1. Mevcut raporda endpoint, kalibrasyon, veri bölmesi ve uygulama tanımlarını düzeltmek.
2. İlk makalenin kapsamını koşula bağlı eğitim kararlılığı ve ampirik karşılaştırma olarak belirlemek.
3. İlgili çalışmaları yalnızca terim adıyla değil, heterojen aktivasyon ataması ve nöron oranı fikri üzerinden taramak.
4. Mevcut oran, MSE ve olumsuz sonuçları da kapsayan yöntem, bulgular ve sınırlılıklar bölümlerini yazmak.
5. Sonraki makale için bağımsız seed’ler, ayrı RNG akışları, aktivasyona uygun başlangıçlandırma, mekanizma ölçümleri ve yoğun LR taraması tasarlamak.

Bu görüşmelerde önerilen kod düzeltmeleri veya yeni deneyler uygulanmamış; yalnızca sonuçlar denetlenmiş, analiz dosyaları oluşturulmuş ve makale kapsamı tartışılmıştır.

## 12. Kaynaklar ve mevcut analiz dosyaları

Terimlerin önceki kullanımı ve mekanizma tartışması için incelenen kaynaklar:

- [Glorot, Bordes ve Bengio (2011), Deep Sparse Rectifier Neural Networks](https://proceedings.mlr.press/v15/glorot11a.html).
- [Glorot ve Bengio (2010), Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html).
- [He ve diğerleri (2015), Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852).
- [Identification of untrained class data using neuron clusters (2023)](https://link.springer.com/article/10.1007/s00521-023-08265-x).
- [Using Summary Layers to Probe Neural Network Behaviour (2020)](https://scielo.org.za/scielo.php?pid=S2313-78352020000200007&script=sci_arttext).

Bu kaynaklar, mevcut hibrit yönteminin özgünlüğüne ilişkin tamamlanmış bir literatür taraması oluşturmaz.

Denetimde üretilen dosyalar:

- [Ayrıntılı deneysel değerlendirme](/Users/khaos/Desktop/article/version5/results_aws/research_audit/evaluation.md)
- [Tüm metrik özetleri](/Users/khaos/Desktop/article/version5/results_aws/research_audit/all_metrics.csv)
- [ReLU’ya göre mutlak ve göreli metrik değişimleri](/Users/khaos/Desktop/article/version5/results_aws/research_audit/baseline_metric_deltas.csv)
- [Eşleştirilmiş karşılaştırmalar ve güven aralıkları](/Users/khaos/Desktop/article/version5/results_aws/research_audit/paired_comparisons.csv)
- [Kalibrasyon denetimi](/Users/khaos/Desktop/article/version5/results_aws/research_audit/calibration_audit.csv)
- [Öğrenme oranı tanı tablosu](/Users/khaos/Desktop/article/version5/results_aws/research_audit/lr_diagnostics.csv)
- [Sınıf bazında metrikler](/Users/khaos/Desktop/article/version5/results_aws/research_audit/class_metrics.csv)
- [Denetim özeti görseli](/Users/khaos/Desktop/article/version5/results_aws/research_audit/audit_overview.png)
