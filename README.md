# Akbank_Brain_Tumor_Project
🧠 Brain Tumor MRI Classification
🎯 Projenin Amacı

Beyin tümörleri, erken teşhis edilmediğinde ölümcül sonuçlara yol açabilen ciddi hastalıklardan biridir. Modern tıpta MRI (Manyetik Rezonans Görüntüleme) en kritik tanı yöntemlerinden biri olsa da, görüntülerin yorumlanması uzmanlık ve zaman gerektirir.

Bu projede, derin öğrenme yöntemlerini kullanarak MRI görüntülerinden beyin tümörlerini otomatik olarak sınıflandırabilen bir yapay zeka modeli geliştirdim.
Amaç, doktorlara yardımcı bir karar destek sistemi sunarak hem teşhis sürecini hızlandırmak hem de insan hatasını en aza indirmektir.

📊 Veri Seti Hakkında

Veri seti, Kaggle üzerinde paylaşılan Brain Tumor MRI Dataset’ten alınmıştır.

Görseller, farklı beyin tümörü tiplerini içermektedir.

İşleme öncesi veriler:

Farklı boyutlarda renkli/gri görsellerden oluşuyordu.

Bu nedenle tüm görüntüler 128×128 piksel, gri tonlamalı olacak şekilde yeniden boyutlandırıldı.

Veriler, %80 eğitim – %20 test şeklinde ayrılmış ve sınıf dağılımı korunmuştur.

📌 Not: Verinin Kaggle üzerinde büyük boyutlu olması nedeniyle, projeyi doğrudan Kaggle ortamında çalıştırdım.

⚙️ Kullanılan Yöntemler ve Adımlar
🔹 Veri Önişleme

İlk adımda, ham MRI görüntülerini modele uygun hale getirdim:

Yeniden boyutlandırma (128×128)

Gri tonlamaya dönüştürme

Normalizasyon (0–255 → 0–1)

Eğitim ve test ayrımı (train_test_split)

Veri görselleştirmeleri: örnek görseller, sınıf dağılımları (countplot)

Bu sayede modelin daha hızlı ve kararlı öğrenebilmesi için veriler standart bir formata kavuştu.

🔹 Data Augmentation (Veri Çoğaltma)

Gerçek hayatta MRI görüntüleri farklı açılardan veya koşullarda çekilebilir. Bu çeşitliliği taklit etmek için data augmentation uyguladım:

Rotation (Döndürme)

Horizontal/Vertical Flip (Yansıma)

Zoom (Yakınlaştırma/Uzaklaştırma)

Color Jitter (Parlaklık/kontrast değişiklikleri)

Böylece model, sadece ezberleyen değil; genelleştirebilen bir yapıya kavuştu.

🔹 Model Mimarisi (CNN)

Modeli sıfırdan CNN tabanlı olacak şekilde tasarladım:

Convolutional Layers (özellik çıkarımı için)

Pooling Layers (boyut küçültme, önemli bilgileri koruma)

Dropout (overfitting’i azaltma)

Dense Layers (Fully Connected) (sınıflandırma için)

Aktivasyon fonksiyonları: ReLU ve Softmax

Optimizer: Adam / SGD

📌 Bonus olarak Transfer Learning denemeleri de yaptım (VGG16, ResNet). Bu modeller, önceden büyük veri setlerinde eğitildikleri için daha iyi başlangıç noktası sundular.

📈 Modelin Değerlendirilmesi

Eğitim sürecinde modeli farklı metriklerle değerlendirdim:

Accuracy & Loss Grafikleri: Eğitim ve doğrulama süreçlerini epoch bazında karşılaştırdım.

Confusion Matrix: Hangi sınıflarda doğru/yanlış tahminler yaptığını inceledim.

Classification Report: Precision, Recall, F1-score metriklerini hesapladım.

Grad-CAM / Eigen-CAM: Modelin, tahmin yaparken MRI görüntüsünün hangi bölgelerine odaklandığını görselleştirdim.

Bu adım, modelin gerçekten anlamlı bölgelerden öğrenip öğrenmediğini kontrol etmek için çok kritik oldu.

🔧 Hiperparametre Optimizasyonu

Modeli daha güçlü hale getirmek için farklı hiperparametrelerle denemeler yaptım:

Katman sayısı

Filtre boyutu ve sayısı

Kernel boyutları

Dropout oranı

Dense layer boyutları

Learning rate

Batch size

Optimizer seçimi

Ekstra olarak:

Keras Tuner ile otomatik optimizasyon denemeleri yaptım.

Overfitting sorununu azaltmak için Dropout ve L2 regularization kullandım.

🏆 Elde Edilen Sonuçlar

Sonuçlar oldukça umut verici çıktı:

Eğitim ve test doğrulukları birbirine yakın seyretti → Overfitting kontrol altında tutuldu.

Doğruluk oranı, veri artırma ve optimizasyon sonrası %XX seviyelerine ulaştı.

Grad-CAM analizleri, modelin karar verirken tümör bölgesine odaklandığını gösterdi.

Bu sonuçlar, modelin gerçek hayatta karar destek sistemi olarak kullanılabilir potansiyele sahip olduğunu ortaya koyuyor.

🔗 Proje Kaynakları

📎 Kaggle Notebook Linki:https://www.kaggle.com/code/mustafaerdogan0001/akbank-brain-tumor-project/edit

📎 
