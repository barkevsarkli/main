# Danışman değerlendirmesi için kısa notlar — 1 Ekim 2026

**Mixed-activation layers: Robustness first, accuracy at width** başlıklı taslak, mevcut bütün deney gruplarını ve 1 Ekim incelemesindeki doğrulanmış ek bulguları kapsar. Yeni eğitim koşusu yapılmadı. Ayrıntılı değişiklik dökümü `revision_2026-10-01.md` içindedir.

En güçlü kontrollü karşılaştırma MNIST H=128, LR=0.03'tür. İncelenen uygulamada aynı tohum, ReLU ve interleave tanh–ReLU için aynı başlangıç ağırlıklarını ve veri sırasını sağlar. Son doğrulama skoru <%15 olan koşular 30/30 ve 0/30; ortalamalar %10.611 ve %96.761'dir. Seçilmiş checkpoint testinde de +0.695 yüzde puanı fark ve 30/30 pozitif tohum farkı vardır. Bu sabit-LR karşılaştırması, ayrı ayrı iyi ayarlanmış modellerin farkı değildir.

Kayıp izleri artık mekanizma tartışmasını destekler: ReLU son CE'si ln(10)'a yakındır ve tahminler neredeyse tek sınıfa iner. Leaky modeller farklı, ani kayıp sıçramaları gösterir. Bunlar ölü katman ve büyük sinyal/ağırlık hipotezleriyle tutarlıdır; gizli aktivasyon kaydı olmadan mekanizma kanıtı değildir.

Dayanıklılık ölçüte bağlıdır. Hiçbir koşunun 10 puandan fazla kaybetmemesi kuralıyla üst test edilmiş LR oranı 1.5× olur. 2.5× sonucu, yüksek kısmi kayıpları göz ardı eden <%15 son skor kuralından gelir. Hibritin en iyi LR'si de iki katına kayar; kendi optimumuna göre normalize edilmiş eğri genel bir aralık genişlemesini desteklemez.

Genişlik sonuçları hem katkıyı hem sınırı gösterir. Kalibrasyon tohumlarında MNIST koruması daha çok genişliğe yayılır, ancak H=512 hibriti %72.843'e düşer. CIFAR H=256 ve 512'de aynı LR'de hibrit de 6/6 çöker. CIFAR'da ilk epochtaki öğrenememe ile MNIST'teki geç çöküş ayrı tutulmalıdır. Saf sigmoid'in bazı sabit-LR koşullarında hibritten daha iyi olduğu da korunmuştur.

Danışmanla öncelikli kararlar:

1. Makalenin katkısını, genişliğe ve eğitim koşuluna bağlı dayanıklılık bulgusu olarak çerçevelemek.
2. Eksik checkpoint uyumlu LR değerlendirmelerini, kaynak–AWS ikilisi eşleşmesini ve MSE türev ölçeğini tamamlamak.
3. Devam çalışmasında oran × LR deneyleri, daha uzun eğitimler, aktivasyon/gradient/ağırlık normu kayıtları ve uygun başlangıç ölçeklerini test etmek. Deterministik maskelerde zaten var olan seed eşleşmesini korumak; random maskeye de genişletmek.

İngilizce metin TMLR'nin isimlerin göründüğü preprint biçimindedir. Kurum bilgileri ve Rahim hocanın e-postası henüz verilmediği için eklenmedi. Nihai gönderim sürümü ayrıca anonimleştirilmeli ve iki yazar tarafından gözden geçirilmelidir.
