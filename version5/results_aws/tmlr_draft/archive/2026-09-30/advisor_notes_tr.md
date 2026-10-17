# Danışman değerlendirmesi için kısa notlar

Bu metin, mevcut sonuçlardan hazırlanan ilk İngilizce makale taslağıdır. Yeni eğitim koşusu yapılmamıştır. Ana dosya `main.tex`, derlenmiş sürüm `main.pdf` olarak verilmiştir.

## Ana iddia

Tanh–ReLU karışımı, incelenen bazı koşullarda eğitimin ilerleyen aşamalarında doğruluğun çökmesini önleyebilir. En güçlü örnek MNIST, H=128, LR=0.03 koşuludur: ReLU 30/30 koşuda %15 son doğrulama doğruluğunun altına inerken, 50:50 tanh–ReLU karışımında bu sayı 0/30'dur. Ortalama son doğrulama doğrulukları %10.611 ve %96.761'dir. Aynı genişlikte karışım ek öğrenilebilir parametre gerektirmez.

Bu, bütün ağlar veya bütün dayanıklılık türleri için genel bir sonuç değildir. Gürültü, saldırı ve dağılım değişimi testleri yapılmamıştır. Aynı sabit öğrenme oranında saf sigmoid'in %98.031 ile hibritten daha yüksek sonuç verdiği de makalede belirtilmiştir.

## Mevcut verilerle söylenebilenler

- MNIST H=128 için kendi en iyi son doğrulama skorunun 1 yüzde puanı içinde kalan test edilmiş LR aralığının üst sınırı ReLU'da 0.01, tanh–ReLU'da 0.02'dir. H=64'te aynı ölçütle genişleme görülmez.
- CIFAR-10 bulguları genel bir karışım üstünlüğünü desteklemez; leaky-ReLU önemli bir karşılaştırma modelidir.
- MNIST H=128'de öğrenme oranı seçimi checkpoint ölçütüne uyarlandığında hibritin test doğruluğu farkı +0.178'den +0.029 yüzde puanına düşer. İkinci farkın güven aralığı sıfırı içerir.
- %25, %50 ve %75 koşuları oran etkisini incelemeye yeterlidir; bu sonuçları yeniden üretmek için aynı koşuların tekrarına gerek yoktur. Ancak bu oranların yüksek LR'daki dayanıklılığını gösteren çapraz deneyler mevcut değildir.
- `activation allocation ratio` ikinci aktivasyona ayrılan nöron oranı olarak tanımlandı. `interleave` yerleşim düzeni için korundu. Karışık aktivasyon veya oran fikrinin ilk kez ortaya atıldığı iddia edilmedi.

## Danışmanla karar verilmesi gereken başlıca konular

1. Katkının “eğitim çöküşüne karşı koşula bağlı dayanıklılık” üzerinden sunulması.
2. Eksik 13 CE yeniden kalibrasyon koşulunun ve değişen bir MSE koşulunun mevcut taslağı tamamlamak için çalıştırılıp çalıştırılmayacağı.
3. Başlangıç ağırlık ölçeği, tek örnekli SGD ve MSE türev ölçeğinin yorumlara etkisi. Mevcut sonuçlar uygulandıkları haliyle raporlandı.
4. Daha ayrıntılı devam çalışmasında oran × öğrenme oranı deneyleri; aktivasyon/gradient kayıtları; ayrı rastgele sayı akışları; uygun başlangıç ölçekleri ve farklı veri bölmeleri.
5. Kaynak kodun AWS ikilisiyle eşleşmesinin doğrulanması, kurum bilgilerinin tamamlanması ve nihai gönderim için anonim sürümün hazırlanması.

Taslak TMLR'nin resmi preprint biçimini kullanır. Bu biçim ve mevcut sonuçların bir makalede düzenlenmiş olması, kabul güvencesi veya tamamlanmış bir gönderim değerlendirmesi anlamına gelmez.
