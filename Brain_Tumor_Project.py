# ============== Bağımlılıklar ==============
import numpy as np # temel sayısal hesaplamalar
import pandas as pd # csv okuma/yazma ve veri çerçeveleri


import os
# ============== Yol Çözümleme ==============
# Proje kökü ve Training/Testing dizinlerini sağlam biçimde bulur
def resolve_data_directories():
    try:
        root_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        root_dir = os.getcwd()

    # 1) Ortam değişkenleri öncelikli
    env_train = os.getenv("TRAIN_DIR")
    env_test  = os.getenv("TEST_DIR")
    if env_train and env_test and os.path.isdir(env_train) and os.path.isdir(env_test):
        return root_dir, env_train, env_test

    # 2) Script dizinine göre
    train_dir = os.path.join(root_dir, "Training")
    test_dir  = os.path.join(root_dir, "Testing")
    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        return root_dir, train_dir, test_dir

    # 3) Çalışma alanı kökü (kullanıcının verdiği path)
    workspace_root = r"D:\\Veri_Bilimi_ve_Makine_Öğrenmesi_2025_100_Günlük_Kamp"
    ws_train = os.path.join(workspace_root, "Training")
    ws_test  = os.path.join(workspace_root, "Testing")
    if os.path.isdir(ws_train) and os.path.isdir(ws_test):
        return workspace_root, ws_train, ws_test

    # 4) Bulunamadıysa yine script'e göre döndür (hata mesajları ileride verilecek)
    return root_dir, train_dir, test_dir

# Global yol değişkenleri
PROJECT_ROOT, TRAIN_DIR_PATH, TEST_DIR_PATH = resolve_data_directories()

# ============== CSV İndeks Üretimi ==============
# Klasörleri CSV indeks dosyalarına dök (yol, etiket)
def generate_index_csv(base_dir, out_csv):
    rows = []
    class_to_idx = {}
    if not os.path.isdir(base_dir):
        return None
    for dirpath, dirnames, filenames in os.walk(base_dir):
        label = os.path.basename(dirpath)
        if not filenames:
            continue
        img_files = [f for f in filenames if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
        if not img_files:
            continue
        if label not in class_to_idx:
            class_to_idx[label] = len(class_to_idx)
        y = class_to_idx[label]
        for fn in img_files:
            rows.append((os.path.join(dirpath, fn), y, label))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["filepath","label","class"])
    # Sınıf sırası tutarlı olsun diye label'ları yeniden eşle
    unique_classes = sorted(df["class"].unique().tolist())
    remap = {c:i for i,c in enumerate(unique_classes)}
    df["label"] = df["class"].map(remap)
    df.to_csv(out_csv, index=False)
    return unique_classes

TRAIN_INDEX_CSV = os.path.join(PROJECT_ROOT, "train_index.csv")
TEST_INDEX_CSV  = os.path.join(PROJECT_ROOT, "test_index.csv")
# CSV indeksleri yoksa üret
if not os.path.exists(TRAIN_INDEX_CSV):
    try:
        _ = generate_index_csv(TRAIN_DIR_PATH, TRAIN_INDEX_CSV)
    except Exception as e:
        print(f"[UYARI] Train CSV oluşturulamadı: {e}")
if not os.path.exists(TEST_INDEX_CSV):
    try:
        _ = generate_index_csv(TEST_DIR_PATH, TEST_INDEX_CSV)
    except Exception as e:
        print(f"[UYARI] Test CSV oluşturulamadı: {e}")

import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # 0:all, 1:INFO, 2:WARNING, 3:ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # bazı gürültülü logları azaltır

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

# ============== Ortam/GPU Bilgisi ==============
# GPU kullanılabilirliğini kontrol edelim (yalnızca bilgi amaçlı)
print("TensorFlow sürümü:", tf.__version__)
print("GPU kullanılabilir mi?", "Evet" if tf.config.list_physical_devices('GPU') else "Hayır")

import os
# print(os.listdir('/kaggle/input/brain-tumor-mri-dataset'))  # Kaggle'a özgü satır devre dışı

import cv2

# ============== Dizin Doğrulama ve Bilgilendirme ==============
# Veri seti dizinlerini doğrula ve bilgi yazdır
def _dir_info(path):
    try:
        return len([f for f in os.listdir(path)])
    except Exception:
        return -1

if not os.path.isdir(TRAIN_DIR_PATH):
    print(f"[HATA] Training klasörü bulunamadı: {TRAIN_DIR_PATH}")
    print("Lütfen dizin yapısını şu şekilde ayarlayın: Training/meningioma, glioma, notumor, pituitary")
if not os.path.isdir(TEST_DIR_PATH):
    print(f"[HATA] Testing klasörü bulunamadı: {TEST_DIR_PATH}")
    print("Lütfen dizin yapısını şu şekilde ayarlayın: Testing/meningioma, glioma, notumor, pituitary")

print(f"Proje kökü: {PROJECT_ROOT}")
print(f"Training dizini: {TRAIN_DIR_PATH} (öğe sayısı: {_dir_info(TRAIN_DIR_PATH)})")
print(f"Testing  dizini: {TEST_DIR_PATH} (öğe sayısı: {_dir_info(TEST_DIR_PATH)})")

# ============== Klasör Tabanlı Yükleme İçin Başlangıç Yolu ==============
# Veri setinin ana klasör yolu (yerel Training dizini)
dataset_base_path = TRAIN_DIR_PATH
print(f"Başlangıç klasör yolu: {dataset_base_path}")

# ============== Basit Klasör Tabanlı Veri Yükleme (Opsiyonel) ==============
def load_data_from_folders(base_path):
    """Belirtilen klasör yolundaki tüm görüntüleri sınıflarına göre yükler.
    Eğer kök dizine ait CSV indeks dosyası varsa onu kullanır.
    """
    X = []
    y = []
    class_names = []
    IMG_SIZE = 128
    
    print("\nVeri seti yükleniyor...")

    # Windows'ta Unicode yol sorunlarına karşı güvenli okuma
    def read_image_safe(image_path, as_gray=True):
        try:
            data = np.fromfile(image_path, dtype=np.uint8)
            flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
            img = cv2.imdecode(data, flag)
            return img
        except Exception:
            return None
    
    # 1) CSV indeks varsa onu kullan
    csv_candidate = TRAIN_INDEX_CSV if os.path.normpath(base_path) == os.path.normpath(TRAIN_DIR_PATH) else TEST_INDEX_CSV
    if os.path.exists(csv_candidate):
        try:
            df = pd.read_csv(csv_candidate)
            class_names = sorted(df["class"].unique().tolist())
            label_map = {c:i for i,c in enumerate(class_names)}
            for _, row in df.iterrows():
                image_path = row["filepath"]
                class_index = int(label_map[row["class"]])
                try:
                    image = read_image_safe(image_path, as_gray=True)
                    if image is not None:
                        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                        X.append(image)
                        y.append(class_index)
                except Exception as e:
                    print(f"Hata: {image_path} görüntüsü yüklenemedi. Hata: {e}")
            return np.array(X), np.array(y), class_names
        except Exception as e:
            print(f"[UYARI] CSV indeks okunamadı ({csv_candidate}). Klasör taramaya geri dönülecek: {e}")

    # 2) CSV yoksa: Klasör tarama
    for dirpath, dirnames, filenames in os.walk(base_path):
        filenames = [f for f in filenames if not f.startswith('.')]
        if any(f.endswith(('.jpg', '.jpeg', '.png', '.bmp')) for f in filenames):
            class_name = os.path.basename(dirpath)
            if class_name not in class_names:
                class_names.append(class_name)
            class_index = class_names.index(class_name)
            print(f"'{class_name}' sınıfı için {len(filenames)} görüntü bulunuyor.")
            for image_name in filenames:
                image_path = os.path.join(dirpath, image_name)
                try:
                    image = read_image_safe(image_path, as_gray=True)
                    if image is not None:
                        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                        X.append(image)
                        y.append(class_index)
                except Exception as e:
                    print(f"Hata: {image_path} görüntüsü yüklenemedi. Hata: {e}")

    return np.array(X), np.array(y), class_names

# Veriyi yükleyelim (opsiyonel demo)
X, y, class_names = load_data_from_folders(dataset_base_path)

# Hata kontrolü
if X.shape[0] == 0:
    print("\n--- HATA: Hiç görüntü yüklenemedi. Dosya yapısını veya yolu kontrol edin. ---")
else:
    print("\nVeri yüklemesi başarılı!")
    print(f"Toplam görüntü sayısı: {len(X)}")
    print(f"Toplam etiket sayısı: {len(y)}")
    print(f"Sınıf isimleri: {class_names}")

    # Veriyi eğitim ve test setlerine ayıralım
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nEğitim verisi şekli:", X_train.shape)
    print("Eğitim etiketi şekli:", y_train.shape)
    print("Test verisi şekli:", X_test.shape)
    print("Test etiketi şekli:", y_test.shape)

    # İlk 25 görüntüyü görselleştirelim
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_train[i], cmap='gray')
        plt.xlabel(class_names[y_train[i]])
    plt.tight_layout()
    plt.show()

    # Sınıf dağılımını inceleyelim
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x=y_train)
    plt.title('Eğitim Verisi Sınıf Dağılımı')
    plt.xlabel('Sınıf')
    plt.ylabel('Sayı')
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)

    plt.subplot(1, 2, 2)
    sns.countplot(x=y_test)
    plt.title('Test Verisi Sınıf Dağılımı')
    plt.xlabel('Sınıf')
    plt.ylabel('Sayı')
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    
    plt.tight_layout()
    plt.show()

    # Veriyi normalize edelim (0-255 arası değerleri 0-1 arasına dönüştürelim)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Etiketleri one-hot encoding formatına dönüştürelim
y_train_categorical = keras.utils.to_categorical(y_train, 10)
y_test_categorical = keras.utils.to_categorical(y_test, 10)

X_train = np.expand_dims(X_train, axis=-1) # (5618, 128, 128, 1)
X_test = np.expand_dims(X_test, axis=-1) # (test_sayısı, 128, 128, 1)

# Veri artırma için ImageDataGenerator kullanalım
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# Veri artırma işlemini eğitim verisine uygulayalım
datagen.fit(X_train)

# Normalizasyon ve one-hot encoding sonrası veri boyutlarını kontrol edelim
print("Normalize edilmiş eğitim verisi şekli:", X_train.shape)
print("One-hot encoded eğitim etiketleri şekli:", y_train_categorical.shape)
 #Temel CNN modeli
def create_cnn_model():
    model = keras.Sequential([
        # İlk evrişim bloğu
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # İkinci evrişim bloğu
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Üçüncü evrişim bloğu
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Sınıflandırıcı
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Modeli oluşturalım
cnn_model = create_cnn_model()

# Model özetini görüntüleyelim
cnn_model.summary()

# === Compile + Train + Plot  ===
import numpy as np, matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# 1) X ve y zaten var (X_train, X_test, y_train, y_test)

X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')
if X_train.max() > 1.5:  # 0-255 ise
    X_train /= 255.0
    X_test  /= 255.0

# Kanal eksikse ekle (grayscale -> (H,W,1))
if X_train.ndim == 3:  # (N,H,W)
    X_train = np.expand_dims(X_train, -1)
    X_test  = np.expand_dims(X_test, -1)

input_shape = X_train.shape[1:]  # (128,128,1) bekliyoruz

# Sınıf sayısını doğru hesapla 
num_classes = len(class_names) if 'class_names' in globals() and len(class_names) > 0 \
              else int(np.max(y_train)) + 1

# Etiketleri one-hot'a doğru boyutla çevir
y_train_categorical = keras.utils.to_categorical(y_train, num_classes)
y_test_categorical  = keras.utils.to_categorical(y_test,  num_classes)

# 2) Augmentation (array tabanlı)
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    horizontal_flip=True
)
datagen.fit(X_train)

# 3) Model: input_shape ve sınıf sayısına göre güncelledik
def create_cnn_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape),                 # (128,128,1) ya da (128,128,3)
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # <-- doğru sınıf sayısı
    ])
    return model

cnn_model = create_cnn_model(input_shape, num_classes)

cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4) Callback'ler
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=8, restore_best_weights=True
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1
)

# 5) Eğitim
history = cnn_model.fit(
    datagen.flow(X_train, y_train_categorical, batch_size=32),
    epochs=25,
    validation_data=(X_test, y_test_categorical),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 6) Accuracy & Loss grafikleri
hist = history.history
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(hist.get('accuracy', []), label='Eğitim Doğruluğu')
plt.plot(hist.get('val_accuracy', []), label='Doğrulama Doğruluğu')
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.plot(hist.get('loss', []), label='Eğitim Kaybı')
plt.plot(hist.get('val_loss', []), label='Doğrulama Kaybı')
plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

plt.tight_layout(); plt.show()

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 1) Test seti performansı
test_loss, test_acc = cnn_model.evaluate(X_test, y_test_categorical, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# 2) Tahminler
y_pred_prob = cnn_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

# 3) Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# 4) Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

correct_indices = np.where(y_pred == y_test)[0]
incorrect_indices = np.where(y_pred != y_test)[0]

plt.figure(figsize=(12, 8))

# Bazı doğru ve yanlış tahmin örneklerini görselleştirelim
correct_indices = np.where(y_pred == y_test)[0]
incorrect_indices = np.where(y_pred != y_test)[0]

plt.figure(figsize=(12, 8))

for i, correct in enumerate(correct_indices[:4]):
    plt.subplot(2, 4, i+1)
    plt.imshow(X_test[correct].squeeze(), cmap="gray") # squeeze = (128,128,1) -> (128,128)
    plt.title(f"Tahmin: {class_names[y_pred[correct]]}\nGerçek: {class_names[y_test[correct]]}")
    plt.axis('off')

# Doğru tahmin örnekleri
for i, correct in enumerate(correct_indices[:4]):
    plt.subplot(2, 4, i+1)
    plt.imshow(X_test[correct])
    plt.title(f"Tahmin: {class_names[y_pred[correct]]}\nGerçek: {class_names[y_test[correct]]}")
    plt.axis('off')

# Yanlış tahmin örnekleri
for i, incorrect in enumerate(incorrect_indices[:4]):
    plt.subplot(2, 4, i+5)
    plt.imshow(X_test[incorrect])
    plt.title(f"Tahmin: {class_names[y_pred[incorrect]]}\nGerçek: {class_names[y_test[incorrect]]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

import os, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from PIL import Image
import matplotlib.pyplot as plt

# ================= KULLANICI AYARLARI =================
# Modelinizin eğitildiği görüntü boyutları ve kanal sayısı
INPUT_H = 128   # Yükseklik (Height)
INPUT_W = 128   # Genişlik (Width)
INPUT_C = 1     # Kanal: Gri tonlama için 1, RGB için 3

# Model özetinizdeki SON Conv2D katmanının adı. (Sizin için "conv2d_14" idi)
LAST_CONV = "conv2d_11" 

# Görüntülerin bulunduğu ana dizin
IMG_DIR = TEST_DIR_PATH 

# Sınıf adlarınız.
class_names = ['meningioma', 'glioma', 'notumor', 'pituitary'] 

# Diğer Ayarlar
COLORMAP = "viridis" 
ALPHA = 0.45 
PREPROCESS = lambda x: x / 255.0 
IMG_PATH = None 
# =======================================================


# ---------------- Yardımcı Fonksiyonlar ----------------
def resolve_model():
    # Modeli cnn_model adıyla bulmaya çalış
    g = globals()
    if 'cnn_model' in g and hasattr(g['cnn_model'], "predict"):
        mdl = g['cnn_model']
        
        # Hata Giderme: Modeli bir dummy input ile çağırarak .input özelliğini zorla oluştur.
        # Bu, Keras'ta sequential modelin yapısını kesinleştiren en agresif yoldur.
        dummy_input = tf.zeros((1, INPUT_H, INPUT_W, INPUT_C)) 
        _ = mdl(dummy_input)
        
        print(f"[INFO] Model: cnn_model bulundu ve yapısı zorla inşa edildi.")
        return mdl
    raise RuntimeError("Eğitilmiş model RAM'de yok (cnn_model). Lütfen modeli eğitin.")

def pick_image_path(img_path, img_dir):
    # Dizin içinden rastgele bir görsel seç
    if img_path and os.path.isfile(img_path): return img_path
    cands = []
    for root,_,files in os.walk(img_dir):
        for f in files:
            if f.lower().endswith((".png",".jpg",".jpeg",".bmp")):
                cands.append(os.path.join(root,f))
    if not cands: raise FileNotFoundError(f"Uygun görsel bulunamadı: {img_dir}")
    return random.choice(cands)


# ---------------- Hazırlık ve Yükleme ----------------
mdl = resolve_model()
img_path = pick_image_path(IMG_PATH, IMG_DIR)
true_label = os.path.basename(os.path.dirname(img_path)) 

# Görsel yükle ve ön işle
if INPUT_C == 1:
    orig = Image.open(img_path).convert("L")
    res  = orig.resize((INPUT_W, INPUT_H), resample=Image.BILINEAR)
    inp  = np.array(res, dtype=np.float32)[..., np.newaxis]
    orig_arr = np.array(orig, dtype=np.uint8) 
else:
    orig = Image.open(img_path).convert("RGB")
    res  = orig.resize((INPUT_W, INPUT_H), resample=Image.BILINEAR)
    inp  = np.array(res, dtype=np.float32)
    orig_arr = np.array(orig, dtype=np.uint8) 

inp_batch = PREPROCESS(np.expand_dims(inp, axis=0)) if PREPROCESS else np.expand_dims(inp, axis=0)


# ---------------- Grad-CAM Hesaplama ----------------

last_conv_layer = mdl.get_layer(LAST_CONV)
classifier_layer = mdl.layers[-1] 

# Modelin giriş tensor'ını, modelin inputs listesinden alıyoruz.
# Bu, .input hatasını çözmenin en güvenilir yoludur.
grad_model_func = tf.keras.models.Model(
    inputs=mdl.inputs[0],
    outputs=[last_conv_layer.output, classifier_layer.output]
)

with tf.GradientTape() as tape:
    conv_out, preds = grad_model_func(inp_batch)
    pred_idx = int(tf.argmax(preds[0]).numpy())
    class_channel = preds[:, pred_idx]
    
grads = tape.gradient(class_channel, conv_out)
pooled = tf.reduce_mean(grads, axis=(0,1,2))
conv_out = conv_out[0]
heat = tf.tensordot(conv_out, pooled, axes=(2,0))
heat = tf.nn.relu(heat)
heat = (heat / (tf.reduce_max(heat) + 1e-8)).numpy().astype(np.float32)

# Orijinal boyuta büyütme ve bindirme
heat_img = Image.fromarray(np.uint8(255*heat))
heat_big = np.array(heat_img.resize((orig.width, orig.height), resample=Image.BILINEAR), dtype=np.float32)/255.0

cmap = plt.cm.get_cmap(COLORMAP)
lut  = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
heat_rgb = lut[np.uint8(255*np.clip(heat_big,0,1))]

if INPUT_C == 1:
    orig_rgb = np.stack([orig_arr, orig_arr, orig_arr], axis=-1)
else:
    orig_rgb = orig_arr
    
overlay = (ALPHA * heat_rgb.astype(np.float32) + orig_rgb.astype(np.float32) * (1-ALPHA)).clip(0,255).astype(np.uint8)

# Tahmin adı
pred_name = (class_names[pred_idx] if class_names and 0 <= pred_idx < len(class_names) else str(pred_idx))


# ---------------- Çizim ----------------
plt.figure(figsize=(15,4))
plt.suptitle(f"Tahmin: {pred_name} (Gerçek: {true_label})", fontsize=16)

plt.subplot(1,3,1)
plt.imshow(orig_arr, cmap="gray" if INPUT_C==1 else None)
plt.title("Orijinal Görüntü")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(heat_big, cmap=COLORMAP)
plt.title("Grad-CAM Isı Haritası")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(overlay)
plt.title("Isı Haritası Bindirilmiş")
plt.axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95]); 
plt.show()

import os, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from PIL import Image
import matplotlib.pyplot as plt

# ================= KULLANICI AYARLARI =================
# Modelinizin eğitildiği görüntü boyutları ve kanal sayısı
INPUT_H = 128   # Yükseklik (Height)
INPUT_W = 128   # Genişlik (Width)
INPUT_C = 1     # Kanal: Gri tonlama için 1, RGB için 3

# Model özetinizdeki SON Conv2D katmanının adı.
# Önceki çıktılara göre doğru değer:
LAST_CONV = "conv2d_11" 

# Görüntülerin bulunduğu ana dizin
IMG_DIR = TEST_DIR_PATH 

# Sınıf adlarınız.
class_names = ['meningioma', 'glioma', 'notumor', 'pituitary'] 

# Diğer Ayarlar
COLORMAP = "viridis" 
ALPHA = 0.45 
PREPROCESS = lambda x: x / 255.0 
IMG_PATH = None 
# =======================================================


# ---------------- Yardımcı Fonksiyonlar ----------------
def resolve_model():
    # Modeli cnn_model adıyla bulmaya çalış
    g = globals()
    if 'cnn_model' in g and hasattr(g['cnn_model'], "predict"):
        mdl = g['cnn_model']
        
        # Hata Giderme: Modeli dummy input ile çağırarak .input özelliğini zorla oluştur.
        dummy_input = tf.zeros((1, INPUT_H, INPUT_W, INPUT_C)) 
        _ = mdl(dummy_input)
        
        print(f"[INFO] Model: cnn_model bulundu ve yapısı zorla inşa edildi.")
        return mdl
    raise RuntimeError("Eğitilmiş model RAM'de yok (cnn_model). Lütfen modeli eğitin.")

def pick_image_path(img_path, img_dir):
    # Dizin içinden rastgele bir görsel seç
    if img_path and os.path.isfile(img_path): return img_path
    cands = []
    for root,_,files in os.walk(img_dir):
        for f in files:
            if f.lower().endswith((".png",".jpg",".jpeg",".bmp")):
                cands.append(os.path.join(root,f))
    if not cands: raise FileNotFoundError(f"Uygun görsel bulunamadı: {img_dir}")
    return random.choice(cands)


# ---------------- Hazırlık ve Yükleme ----------------
mdl = resolve_model() # Model burada yükleniyor
img_path = pick_image_path(IMG_PATH, IMG_DIR)
true_label = os.path.basename(os.path.dirname(img_path)) 

# Görsel yükle ve ön işle
if INPUT_C == 1:
    orig = Image.open(img_path).convert("L")
    res  = orig.resize((INPUT_W, INPUT_H), resample=Image.BILINEAR)
    inp  = np.array(res, dtype=np.float32)[..., np.newaxis]
    orig_arr = np.array(orig, dtype=np.uint8) 
else:
    orig = Image.open(img_path).convert("RGB")
    res  = orig.resize((INPUT_W, INPUT_H), resample=Image.BILINEAR)
    inp  = np.array(res, dtype=np.float32)
    orig_arr = np.array(orig, dtype=np.uint8) 

inp_batch = PREPROCESS(np.expand_dims(inp, axis=0)) if PREPROCESS else np.expand_dims(inp, axis=0)


# ---------------- Eigen-CAM Hesaplama ----------------

last_conv_layer = mdl.get_layer(LAST_CONV)

# 1. Sadece Conv katmanının çıktısını veren bir model oluştur.
eigen_model = tf.keras.models.Model(
    inputs=mdl.inputs[0],
    outputs=last_conv_layer.output
)

# Son Conv katmanının çıktılarını (aktivasyon haritalarını) al
conv_out = eigen_model.predict(inp_batch)[0] 

# 2. Boyut İndirgeme (SVD)
# (h, w, k) -> (h*w, k) olarak yeniden şekillendir
reshaped_out = tf.reshape(conv_out, [-1, conv_out.shape[-1]]) 
s, u, v = tf.linalg.svd(reshaped_out)

# 3. En Büyük Bileşeni Seç (Principal Component)
eigen_weights = v[:, 0]

# 4. Isı Haritasını Oluştur (Eigen-CAM)
heat = tf.tensordot(conv_out, eigen_weights, axes=(2, 0))

# ReLU ve Normalizasyon uygula
heat = tf.nn.relu(heat)
heat = (heat / (tf.reduce_max(heat) + 1e-8)).numpy().astype(np.float32)

# Orijinal boyuta büyütme ve bindirme
heat_img = Image.fromarray(np.uint8(255*heat))
heat_big = np.array(heat_img.resize((orig.width, orig.height), resample=Image.BILINEAR), dtype=np.float32)/255.0

cmap = plt.cm.get_cmap(COLORMAP)
lut  = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
heat_rgb = lut[np.uint8(255*np.clip(heat_big,0,1))]

if INPUT_C == 1:
    orig_rgb = np.stack([orig_arr, orig_arr, orig_arr], axis=-1)
else:
    orig_rgb = orig_arr
    
overlay = (ALPHA * heat_rgb.astype(np.float32) + orig_rgb.astype(np.float32) * (1-ALPHA)).clip(0,255).astype(np.uint8)

# Tahmin adı
preds = mdl.predict(inp_batch)
pred_idx = int(tf.argmax(preds[0]).numpy())
pred_name = (class_names[pred_idx] if class_names and 0 <= pred_idx < len(class_names) else str(pred_idx))


# ---------------- Çizim ----------------
plt.figure(figsize=(15,4))
plt.suptitle(f"Eigen-CAM Sonuçları | Tahmin: {pred_name} (Gerçek: {true_label})", fontsize=16)

plt.subplot(1,3,1)
plt.imshow(orig_arr, cmap="gray" if INPUT_C==1 else None)
plt.title("Orijinal Görüntü")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(heat_big, cmap=COLORMAP)
plt.title("Eigen-CAM Isı Haritası")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(overlay)
plt.title("Isı Haritası Bindirilmiş")
plt.axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# ===================== RESUMABLE RANDOM SEARCH (Brain Tumor MRI) =====================
import os, json, pickle, random, numpy as np, warnings, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # INFO/WARNING'i sustur
import tensorflow as tf
tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
tf.config.optimizer.set_jit(False)
tf.get_logger().setLevel("ERROR"); warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt

# --------- Random Search Sabitleri ---------
DATA_ROOT = PROJECT_ROOT   # Yerel proje kökü
TRAIN_DIR = os.path.join(DATA_ROOT, "Training")
TEST_DIR  = os.path.join(DATA_ROOT, "Testing")

IMG_SIZE = (128, 128)       # model giriş boyutu (H, W)
CHANNELS = 1                # MRI gri ise 1; RGB istiyorsan 3 yap
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], CHANNELS)

EPOCHS  = 15                # hız için düşürdüm; ES zaten keser
TRIALS  = 10                # toplam deneme hedefi
SEED    = 42
AUTOTUNE = tf.data.AUTOTUNE
tf.random.set_seed(SEED); np.random.seed(SEED); random.seed(SEED)

STATE_PATH = "state.json"   # kaldığı yer/state bilgisi
BEST_MODEL_PATH = "best_model.keras"

# --------- 0) Veri Yükleme ---------
def get_datasets(img_size=IMG_SIZE, batch_size=32, channels=CHANNELS):
    """Unicode yol sorunlarına takılmadan CSV indekslerinden tf.data pipeline kurar."""
    if not os.path.exists(TRAIN_INDEX_CSV):
        raise FileNotFoundError(f"Train CSV yok: {TRAIN_INDEX_CSV}")
    if not os.path.exists(TEST_INDEX_CSV):
        raise FileNotFoundError(f"Test CSV yok: {TEST_INDEX_CSV}")

    df_train_full = pd.read_csv(TRAIN_INDEX_CSV)
    df_test = pd.read_csv(TEST_INDEX_CSV)

    class_names = sorted(df_train_full["class"].unique().tolist())
    class_to_idx = {c:i for i,c in enumerate(class_names)}
    num_classes = len(class_names)

    # Stratified split train/val
    train_df, val_df = train_test_split(
        df_train_full,
        test_size=0.2,
        random_state=SEED,
        stratify=df_train_full["class"],
        shuffle=True
    )

    def build_dataset_from_df(df, shuffle=False):
        filepaths = df["filepath"].astype(str).values
        labels = df["class"].map(class_to_idx).astype(np.int64).values

        def _reader_py(path_bytes):
            # path_bytes: np.bytes_
            path = path_bytes.decode("utf-8", errors="ignore")
            data = np.fromfile(path, dtype=np.uint8)
            flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
            img = cv2.imdecode(data, flag)
            if img is None:
                # Bozuk görsellerde siyah görsel döndür
                ch = 1 if channels == 1 else 3
                return np.zeros((img_size[0], img_size[1], ch), dtype=np.uint8)
            # Resize
            img = cv2.resize(img, (img_size[1], img_size[0]))
            if channels == 1 and img.ndim == 2:
                img = img[..., np.newaxis]
            if channels == 3 and img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            return img

        ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

        def _map_fn(path, label):
            img = tf.numpy_function(_reader_py, [path], Tout=tf.uint8)
            img = tf.ensure_shape(img, [img_size[0], img_size[1], channels])
            img = tf.cast(img, tf.float32) / 255.0
            label = tf.cast(label, tf.int32)
            return img, label

        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).cache().prefetch(AUTOTUNE)
        return ds

    train_ds = build_dataset_from_df(train_df, shuffle=True)
    val_ds   = build_dataset_from_df(val_df, shuffle=False)
    test_ds  = build_dataset_from_df(df_test, shuffle=False)

    # Class weights hesapla (train set üzerinden)
    counts_series = train_df["class"].value_counts().reindex(class_names).fillna(0).astype(int)
    counts = counts_series.values
    total = counts.sum()
    class_weights = {i: float(total/(num_classes*max(counts[i],1))) for i in range(num_classes)}
    print("[INFO] class_names:", class_names)
    print("[INFO] class_counts:", dict(zip(class_names, counts.tolist())))
    print("[INFO] class_weights:", class_weights)

    return train_ds, val_ds, test_ds, num_classes, class_names, class_weights

# --------- 1) Model Kurucu ---------
def build_model(input_shape=INPUT_SHAPE,
                num_classes=4,
                conv_blocks=3,
                base_filters=32,
                kernel_size=3,
                dropout=0.3,
                dense_units=128,
                l2_weight=1e-4,
                optimizer_name="adam",
                lr=1e-3,
                use_bn=True,
                augment=True):
    wd = regularizers.l2(l2_weight)
    inp = keras.Input(shape=input_shape)
    x = inp

    if augment:
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.05)(x)
        x = layers.RandomZoom(0.10)(x)

    filters = base_filters
    for _ in range(conv_blocks):
        x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=not use_bn,
                          kernel_regularizer=wd)(x)
        if use_bn: x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=not use_bn,
                          kernel_regularizer=wd)(x)
        if use_bn: x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout)(x)
        filters = min(filters*2, 256)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=wd)(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)  # mixed_precision için güvenli

    model = keras.Model(inp, out)

    opt_name = optimizer_name.lower()
    if opt_name == "adam":
        opt = keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == "adamw":
        opt = keras.optimizers.AdamW(learning_rate=lr, weight_decay=l2_weight)
    elif opt_name == "rmsprop":
        opt = keras.optimizers.RMSprop(learning_rate=lr)
    else:
        opt = keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# --------- 2) HP Alanı ---------
HP_SPACE = {
    "conv_blocks":   [2,3,4],
    "base_filters":  [16,32,64],
    "kernel_size":   [3,5],
    "dropout":       [0.2,0.3,0.4,0.5],
    "dense_units":   [64,128,256],
    "l2_weight":     [1e-5,1e-4,5e-4,1e-3],
    "optimizer":     ["adam","adamw","rmsprop"],
    "lr":            [1e-4, 3e-4, 1e-3, 2e-3],
    "batch_size":    [16,32,64],
    "use_bn":        [True],
    "augment":       [True, False]
}

# --------- 3) State yükle/başlat ---------
state = {"completed_trials": 0, "best_val_acc": -1.0, "best_hp": None, "best_batch": 32}
if os.path.exists(STATE_PATH):
    try:
        with open(STATE_PATH, "r") as f:
            saved = json.load(f)
            state.update(saved)
        print(f"[RESUME] Found state: {state}")
    except Exception as e:
        print("[WARN] state.json okunamadı, sıfırdan başlayacak:", e)

# Veri pipeline (tek kez kuruluyor)
train_ds, val_ds, test_ds, num_classes, class_names, class_weights = get_datasets(
    img_size=IMG_SIZE, batch_size=32, channels=CHANNELS
)

# --------- 4) Random Search (kaldığı yerden) ---------
start_t = state["completed_trials"] + 1
end_t   = TRIALS

for t in range(start_t, end_t+1):
    hp = {k: random.choice(v) for k,v in HP_SPACE.items()}
    print(f"\n[Trial {t}/{TRIALS}] HP: {hp}")

    # re-batch: hp'deki batch_size'i gerçekten uygula
    bs = hp["batch_size"]
    train_b = train_ds.unbatch().batch(bs).prefetch(AUTOTUNE)
    val_b   = val_ds.unbatch().batch(bs).prefetch(AUTOTUNE)

    model = build_model(
        input_shape=INPUT_SHAPE,
        num_classes=num_classes,
        conv_blocks=hp["conv_blocks"],
        base_filters=hp["base_filters"],
        kernel_size=hp["kernel_size"],
        dropout=hp["dropout"],
        dense_units=hp["dense_units"],
        l2_weight=hp["l2_weight"],
        optimizer_name=hp["optimizer"],
        lr=hp["lr"],
        use_bn=hp["use_bn"],
        augment=hp["augment"]
    )

    es   = keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    rlrop= keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    csv  = keras.callbacks.CSVLogger(f"trial_{t}.csv", append=False)
    ckpt = keras.callbacks.ModelCheckpoint(f"trial_{t}_best.keras", monitor="val_accuracy",
                                           save_best_only=True, mode="max")

    history = model.fit(
        train_b,
        validation_data=val_b,
        epochs=EPOCHS,
        callbacks=[es, rlrop, csv, ckpt],
        class_weight=class_weights,
        verbose=1
    )

    # trial çıktılarını kaydet
    with open(f"trial_{t}.pkl", "wb") as f: pickle.dump(history.history, f)
    with open(f"trial_{t}_hp.json", "w") as f: json.dump(hp, f)

    val_acc = float(np.max(history.history["val_accuracy"]))
    print(f"  --> best val_accuracy: {val_acc:.4f}")

    # en iyiyi güncelle
    if val_acc > state["best_val_acc"]:
        state["best_val_acc"] = val_acc
        state["best_hp"] = hp
        state["best_batch"] = bs
        # trial checkpoint'ini "best_model.keras" olarak kopyala
        try:
            # bazı ortamlarda doğrudan kaydetmek daha güvenli:
            model.save(BEST_MODEL_PATH)
        except Exception:
            pass
        print("[BEST] Updated best model & HP.")

    # state'i güncelle (trial tamamlandı)
    state["completed_trials"] = t
    with open(STATE_PATH, "w") as f: json.dump(state, f)
    print(f"[STATE] Saved: {state}")

# --------- 5) En iyi modeli test et + grafik çiz ---------
print("\n=== EN İYİ SONUÇ (VAL) ===")
print("Completed trials:", state["completed_trials"])
print("Best val_acc:", state["best_val_acc"])
print("Best HP:", state["best_hp"])

# test seti için en iyi batch ile re-batch
best_bs = state.get("best_batch", 32)
test_b  = test_ds.unbatch().batch(best_bs).prefetch(AUTOTUNE)

best_model = None
if os.path.exists(BEST_MODEL_PATH):
    try:
        best_model = keras.models.load_model(BEST_MODEL_PATH)
        test_loss, test_acc = best_model.evaluate(test_b, verbose=0)
        print("Test Acc:", float(test_acc), "| Test Loss:", float(test_loss))
    except Exception as e:
        print("[WARN] Best model yüklenemedi:", e)

# Son trial'in (ya da en iyi trial'in) grafiğini çiz
# En iyi trial'in pkl dosyasını bulmaya çalış
best_hist = None
if state["best_hp"] is not None:
    # best trial'i tahmin etmek için state dosyasından logları tara
    # (pratikçe son trial grafiğini gösterelim; best grafiği istersen dosyadan yükle)
    last_t = state["completed_trials"]
    try:
        with open(f"trial_{last_t}.pkl","rb") as f:
            best_hist = pickle.load(f)
    except Exception:
        pass

if best_hist is not None:
    h = best_hist
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(h["loss"], label="train"); plt.plot(h["val_loss"], label="val")
    plt.title("Loss"); plt.legend(); plt.grid(True)
    plt.subplot(1,2,2); plt.plot(h["accuracy"], label="train"); plt.plot(h["val_accuracy"], label="val")
    plt.title("Accuracy"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()
else:
    print("[INFO] Grafiği göstermek için ilgili trial_*.pkl bulunamadı.")


# Model başarıyla diske kaydedilir
cnn_model.save('best_brain_tumor_model.keras')
