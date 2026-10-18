# version6 — AWS çalıştırma kılavuzu (tek makine, c8g.48xlarge)

version5'teki düzenin aynısı: Graviton4, Ubuntu 24.04 ARM64, spot, eu-central-1. Bütün aşamalar (0–3) bu tek makinede, tek derlemeyle koşar (DECISIONS 42).

**Beklenen süre ve maliyet:** ~14 saat (benchmark sonrası netleşecek). Spot fiyatıyla ~$2,6/saat → ~$37; on-demand ~$110.

---

## 0) Mac'te — yalnızca bir kez

```bash
cd ~/Desktop/article
git status                      # yalnızca version5/CLAUDE_CODE_PROMPT_version6_CNN.md takip dışı olabilir
git push origin v6-cnn
git rev-parse HEAD              # bu hash ile ...
git ls-remote --heads origin v6-cnn   # ... bu hash aynı olmalı
```

Push yapılmadan sunucuya en son düzeltmeler (tohum setleri, bu kılavuz, `run_all_aws.sh`) inmez.

## 1) AWS konsolu — instance'ı aç

1. **Region:** sağ üstten **eu-central-1 (Frankfurt)** seç.
2. **Spot fiyatı:** EC2 → *Spot Requests* → *Pricing history* → Instance type `c8g.48xlarge`, Linux/UNIX. En ucuz Availability Zone'u not et (ör. eu-central-1a).
3. **Kota:** *Service Quotas* → *Amazon EC2*.
   - Spot için *All Standard (A, C, D, H, I, M, R, T, Z) Spot Instance Requests* ≥ 192 vCPU olmalı.
   - On-demand için *Running On-Demand Standard … instances* ≥ 192 olmalı.
   - version5'te çalıştığı için büyük ihtimalle yeterli.
4. **Bütçe alarmı (önerilir):** *Billing* → *Budgets* → aylık maliyet bütçesi, örneğin $80, e-posta uyarısıyla.
5. **Launch instance:**
   - **Name:** `v6-sweep`
   - **AMI:** *Ubuntu Server 24.04 LTS*, mimari **64-bit (Arm)**
   - **Instance type:** `c8g.48xlarge` (192 vCPU, 384 GiB)
   - **Key pair:** `sweep-key` (Mac'te `~/.ssh/sweep-key.pem`)
   - **Network settings → Edit:** 2. adımda seçtiğin AZ'nin subnet'i. Security group: yalnızca **SSH 22**, kaynak **My IP**.
   - **Storage:** 100 GiB **gp3**
   - **Advanced details → Purchasing option:** *Spot instances* → *Customize* → Request type **Persistent**, Interruption behavior **Stop**.
   - **Launch instance**.
6. *Instances* → makineyi seç → **Public IPv4 address**'i kopyala. Aşağıda `<IP>` diye geçiyor. Spot kesintisinden sonra IP değişebilir; değişmesin istersen bir Elastic IP bağla.

## 2) Bağlan ve kur

```bash
# Mac'te
chmod 400 ~/.ssh/sweep-key.pem
ssh -i ~/.ssh/sweep-key.pem ubuntu@<IP>
```

```bash
# Sunucuda
sudo apt update && sudo apt install -y build-essential cmake git python3 tmux htop
nproc            # 192
uname -m         # aarch64
free -g | head -2
cmake --version  # >= 3.20
```

## 3) Kodu indir ve derle

```bash
git clone https://github.com/barkevsarkli/main.git article     # veri setleri de gelir
cd article && git checkout v6-cnn && git log -1 --oneline        # Mac'teki hash ile aynı olmalı
cd version6
./build.sh test                                  # beklenen: 100% tests passed (6/6)
grep -E "GIT_HASH|DIRTY|PLATFORM|FLAGS" build/generated/provenance.h   # DIRTY "0", Linux-aarch64
```

**Önemli:** Bu derleme bütün aşamalarda kullanılacak. Benchmark'tan sonra **yeniden derleme yapma**. Benchmark bazı takipli dosyaları değiştiriyor; yeniden derlersen `DIRTY "1"` olur ve `run_all_aws.sh` başlamayı reddeder.

## 4) Koşmadan önce doğrulama (~15 dk)

```bash
python3 tests/test_selection.py   # test_selection: PASS
python3 tests/regress_v5.py       # version5 ile bit düzeyinde aynılık: hepsi identical
python3 tests/test_runs.py        # RNG hizası, frozen, belirlenimcilik, işçi sayısı, stats değişmezliği
python3 sweep/run_jobs.py results/smoke/jobs.jsonl --workers 192 --results /tmp/aws_smoke
                                  # son satır: "ok 108, failed 0"
```

Bunlardan biri başarısız olursa **dur**; deney başlatma.

## 5) Benchmark (~30–45 dk)

```bash
tmux new -s v6          # SSH koparsa iş devam eder; geri dönmek için: tmux attach -t v6
cd ~/article/version6
python3 tools/bench.py --workers-list 96,144,192 --max-hours 1
python3 tools/estimate.py
cp -r results/bench results/bench_aws
```

Bakılacaklar:

- `results/bench/bench.md`: seçilen işçi sayısı büyük ihtimalle 192.
- **AWS referans oranı ≈ 1,0 olmalı.** Bu, version5'in aynı makinedeki hızının tekrar ölçülmesi.
- `estimate.md`'deki "here (h)" sütunu bu makinede aşama başına süreyi verir. Toplam ~14 saat bekleniyor.
- Toplam **> 25 saat** çıkarsa dur ve planı gözden geçir.

## 6) Bütün aşamaları başlat

tmux içinde:

```bash
cd ~/article/version6
W=192 bash sweep/run_all_aws.sh
```

Script şu sırayla gider ve ilk hatada durur:

1. Stage 0 (240 koşu, ~5 dk)
2. Stage 1 (4.554 koşu, ~4 sa) → `select_stage2.py`
3. Stage 2 taraması (6.860 koşu, ~5 sa) → `select_lr.py`
4. Stage 2 uzatma (≤ 5.880 koşu, ~4–5 sa)
5. Stage 3 (450 koşu, ~0,5 sa)

Her aşamanın sonunda sonuçlar birleştirilir (`results/<stage>/<stage>.csv`) ve diagnostikler `~/v6_archives/stats_<stage>.tar.gz` olarak arşivlenir.

tmux'tan ayrılmak için `Ctrl+B`, ardından `D`.

## 7) İzleme (ayrı bir SSH oturumunda)

```bash
tail -f ~/article/version6/results/run_all_aws.log      # her biten koşu bir satır; 5 dakikada bir özet ve ETA
htop                                                     # 192 çekirdek ~%100
free -g                                                  # swap kullanılmamalı
ls ~/article/version6/results/stage1/rows | wc -l        # biten koşu sayısı
```

**Spot kesintisi ya da script'in durması:**

1. Makine yeniden açılınca bağlan.
2. `tmux new -s v6`, ardından `cd ~/article/version6 && W=192 bash sweep/run_all_aws.sh` çalıştır.
3. Script bitmiş koşuları atlar ve kaldığı yerden devam eder.
4. Hata mesajı varsa `results/<stage>/failures.log` dosyasına bak.

## 8) Sonuçları Mac'e al

Log'da `ALL STAGES COMPLETE` görününce, Mac'te:

```bash
cd ~/Desktop/article/version6
rsync -az --progress -e "ssh -i ~/.ssh/sweep-key.pem" \
  --exclude 'rows/' --exclude 'stats/' \
  ubuntu@<IP>:~/article/version6/results/ results/
mkdir -p ~/Desktop/v6_archives
rsync -az --progress -e "ssh -i ~/.ssh/sweep-key.pem" \
  ubuntu@<IP>:~/v6_archives/ ~/Desktop/v6_archives/
```

- Arşivler toplam birkaç GB. Git'e değil; Drive'a veya S3'e kopyala.
- İstersen satır dosyalarını da ayrıca çekebilirsin: `--exclude 'rows/'` satırını kaldırman yeterli. Ama birleştirilmiş CSV'ler aynı bilgiyi içeriyor.

Ardından Mac'te:

```bash
cd ~/Desktop/article
git add version6/results && git commit -m "version6: AWS results, stages 0-3" && git push origin v6-cnn
```

Analiz (Mac'te; numpy, pandas, scipy, matplotlib gerekir):

```bash
cd ~/Desktop/article/version6
for s in stage0 stage1 stage2 stage3; do python3 analysis/summarise.py results/$s; python3 analysis/figures.py results/$s; done
python3 analysis/summarise.py results/stage2 --seeds 11-30      # önceden kaydedilmiş kesin testler
```

## 9) Kapat — unutma

Persistent spot isteği iptal edilmezse makine kendini yeniden açar. Sıra önemli:

1. EC2 → *Spot Requests* → isteği seç → **Cancel request**. "Terminate instances" kutusunu işaretle.
2. EC2 → *Instances*: makine *terminated* değilse **Terminate** et.
3. EC2 → *Volumes*: 100 GiB disk kaldıysa **Delete** et.
4. Elastic IP bağladıysan **Release** et.
5. Birkaç gün sonra *Billing* → *Bills* üzerinden maliyeti kontrol et.
