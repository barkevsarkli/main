# version6 — deney öncesi bağımsız kontrol (18 Ekim 2026)

**Kontrol edilen:** `v6-cnn` dalı, commit `f49ddbb` (origin ile eşit).  
**Ortam:** Linux aarch64 VM (Ubuntu 22.04, GCC 11.4, glibc 2.35). AWS Graviton ile aynı mimari; ajanın test ettiği macOS/AppleClang'dan farklı bir derleyici ve libm.

## Durum güncellemesi (aynı gün)

A (tohum setleri) ve C (`.cache` temizliği) uygulandı; ajanın açık soruları yanıtlandı (DECISIONS 42–51, PLAN.md *Seed sets*).

## Karar

Kod doğru çalışıyor ve deneylere hazır. Başlamadan önce **bir metodolojik düzeltme (A)** öneriyorum. Ayrıca iki küçük temizlik (C, D) var. Hiçbiri yeni koşu gerektirmiyor.

## Bağımsız olarak doğrulananlar

| # | Kontrol | Sonuç |
|---|---|---|
| 1 | Temiz derleme (Release, `-O3 -march=native -ffp-contract=off`) | Uyarısız |
| 2 | C++ test paketi (`ctest`) | 6/6 geçti |
| 3 | version5 ↔ version6 `mlp2 --rng-mode legacy`, aynı derleyici ve bayraklar | MNIST H=12 relu ve tanh+relu: bütün sütunlar birebir aynı |
| 4 | **Bağımsız NumPy referansı** (ajanın kodu kullanılmadı; float64, 80–150 SGD adımı, `--dump-init`/`--dump-final`) | random / bank_init / frozen; K 3/5/7/9; pool 1/2/4; relu, tanh, leaky-ReLU, sigmoid ve karışımlar; MNIST + CIFAR-10. Göreli ağırlık farkı ≤ 5e-15 |
| 5 | Kernel bankaları (48 JSON) | Her dosyada gain × ölçülen std = τ; K=3 Sobel klasik [[-1,0,1],[-2,0,2],[-1,0,1]] |
| 6 | Belirlenimcilik, diagnostik açık/kapalı, mmap/ifstream | Satırlar birebir aynı (`seconds` hariç) |
| 7 | `run_jobs.py` | 4 iş paralel çalıştı; tekrar çağrıda bitmiş işleri atladı |
| 8 | İş dosyaları ve PROMPT karşılaştırması | Stage 0 = 240, Stage 1 = 4.554, Stage 2 = 6.860 + 5.880, Stage 3 = 450; LR ızgaraları doğru; H=512 yok |

Not 1: NumPy referansında LR=0,3'te fark adım adım büyüyor (60 adımda 1e-15'ten 1e-10'a). Bu, yuvarlama farkının kaotik dinamikte büyümesi; hata değil. LR=0,01'de büyümüyor.

Not 2: leaky-ReLU eğimi kodda float32 olarak 0,1 (yani 0,10000000149). Bu version5 ile aynı.

### Mekanizma ön testi (tek tohum; split RNG; 40 epoch; MNIST H=128, LR=0,03; mlp2)

| | ReLU | tanh+ReLU |
|---|---|---|
| Son val doğruluğu | %9,24 (zirve %96,32, epoch 14) | %96,92 |
| Son eğitim kaybı | 2,3081 (≈ ln 10) | 0,044 |
| hidden2 ölü ReLU birimi (epoch 0 / 10 / 20 / 25) | 0 / 1 / 13 / **128 (hepsi)** | 0 / 0 / 0 / 1 |
| Girdiler arası logit std (epoch 20 → 22) | 38,6 → **3,8e-09** (çıktı girdiden bağımsız) | – |
| Tahmin edilen sınıf sayısı | 10 → **1** | 10 |
| Clip oranı | 0 (clip hiç devreye girmiyor) | – |
| hidden1 / hidden2 ağırlık normu (epoch 0 → 22) | 6,5 → 129 / 6,5 → 100 | – |

Bu tek koşu, makaledeki "ölü ReLU katmanı" yorumunu doğrudan destekliyor ve diagnostiklerin çalıştığını gösteriyor. Kesin sonuç için Stage 0'ın 10 tohumu gerekiyor.

## A. Başlamadan önce düzeltilmeli: tohum çakışması ve seçim yanlılığı

`split` RNG'de aynı tohum aynı başlangıç ağırlıkları ve aynı veri sırası demek. Bu yüzden:

1. **Stage 1 → Stage 2:** Stage 1 tohum 1–3 kullanıyor; Stage 2 taraması tohum 1–10. H=128'de, ortak LR'larda, seçilen konfigürasyonların tohum 1–3 koşuları Stage 1'deki koşuların **birebir aynısı**. Yani kernel'ın seçildiği veri Stage 2 sonuçlarının içinde yeniden yer alıyor. PLAN.md "Stage 2 kazananları yeni tohumlarla tekrar test eder" diyor, ama 1–3 yeni değil.
2. **Stage 2 stress LR:** Stress LR, ReLU'nun tohum 1–10'daki son skorundan seçiliyor. Uzatma 11–30'u ekliyor, ama 30 tohumlu analiz 1–10'u da içeriyor. Böylece ReLU'nun o LR'da çökmesi kısmen seçimle garanti edilmiş oluyor.
3. **Stage 3:** Tohum 1–15, Stage 2 ile çakışıyor. LR'lar da aynı ReLU koşularından seçilmiş.

**Öneri (maliyeti yok):**

- Stage 1 tohumlarını **101–103** yap; version5'in kalibrasyon tohumu geleneği de buydu. Değişecek yerler: `make_jobs.py` içinde `stage1(seeds=...)`, `select_stage2.py` içinde `SEEDS`, PLAN.md ve test_selection.py içindeki tohum varsayımları.
- Stage 3 tohumlarını **31–45** yap. DECISIONS 33'teki tekrarlanan koşular da böylece ortadan kalkar.
- Kesin (confirmatory) testleri seçilmiş LR ve stress LR'da **yalnızca tohum 11–30** üzerinde yapmayı PLAN.md'de önceden kaydet; 1–30 analizi ikincil olsun.

## B. Platformlar arası farklılık — kural: tek platform

Mac'te (AppleClang, macOS libm) üretilen smoke satırlarını Linux GCC'de yeniden koştum. Bazıları birebir aynı çıktı (ör. frozen sobel K=3, mlp2). Bazıları **ilk epoch'tan itibaren** farklı (ör. frozen gauss K=3, mlp1; fark %0,1–0,3 düzeyinde). Sebep büyük ihtimalle tanh/exp'in iki libm'de son bitte farklı sonuç vermesi. DECISIONS 38'deki "üç derleme aynı" gözlemi tek bir iş için doğru, genel değil.

**Kural:**

- Bütün aşamaları (Stage 0 dahil) **aynı AWS AMI, aynı instance tipi ve tek bir derleme** ile koş.
- Mac smoke/bench sonuçlarını analizde kullanma.
- Spot kesintisinde aynı tip ve AMI ile yeniden başlat.

## C–G. Küçük konular

- **C.** `version6/.cache/clangd/` (31 dosya) git'e girmiş. `.gitignore`'a ekle ve `git rm --cached -r version6/.cache` çalıştır.
- **D.** `version5/CLAUDE_CODE_PROMPT_version6_CNN.md` takip edilmeyen bir dosya olarak duruyor. Kopyası `version6/PROMPT.md`'de var; silinebilir.
- **E. AWS hazırlığı:**
  - Veri setleri git'te, yani `git clone` ile geliyor; `scp` gerekmez.
  - Gerekenler: `sudo apt install -y cmake g++ git python3 tmux`, ardından `./build.sh test`.
  - Diagnostik dosyaları toplam ~3 GB; git'e değil, `rsync`/S3 ile taşı.
  - Analiz (pandas, scipy, matplotlib) laptopta yapılabilir; AWS'de yalnızca Python standart kütüphanesi gerekiyor.
- **F. Açık kararlar (DECISIONS):**
  - `--abort-on-collapse`: kapalı kalsın. AWS maliyeti zaten düşük ve çöküş eğrileri analiz için gerekli.
  - Gabor K=3 için λ=2,5: kabul edilebilir; makalede belirtilmeli.
  - version5'in AWS sonuçları FMA kısaltmalı bir derlemeden geliyor: makalede bir cümleyle belirtilmeli.
- **G.** Windows kod yolu hiç derlenmedi. AWS kullanılacaksa önemsiz.

## Süre

Ajanın benchmark'ı tüm planı AWS'de (c8g.48xlarge, 192 işçi) **~14 saat** olarak tahmin ediyor: spot ≈ $37, on-demand ≈ $110. Daha önce verdiğim ~43–50 saatlik tahmin fazla karamsardı; version6'nın vektörlenmiş conv/dense kodu benim varsaydığımdan 3–4 kat hızlı. Başlamadan önce AWS'de 20 dakikalık `tools/bench.py` ile teyit edilmeli.
