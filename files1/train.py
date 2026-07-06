"""
train.py
Training model klasifikasi Batik Aceh (Pucok Rebung, Pinto Aceh, Rencong)
menggunakan Transfer Learning VGG16 (fully frozen) + custom classifier head.
"""

import json
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# ======================================================
# KONFIGURASI
# ======================================================
DATASET_DIR = r"E:\Batik3dataset\dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 123
EPOCHS = 15
NUM_CLASSES = 3
MODEL_OUTPUT_PATH = "batik_aceh_vgg16.keras"
CLASS_NAMES_PATH = "class_names.json"

# ======================================================
# 1. DATA PIPELINE
# ======================================================
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

class_names = train_dataset.class_names
print(f"Kelas terdeteksi: {class_names}")

# Simpan urutan nama kelas agar konsisten dipakai saat inferensi di app.py
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)

# Augmentasi ringan HANYA untuk data training agar motif batik tidak terdistorsi
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(15 / 360),  # rotasi maksimal 15 derajat
    ],
    name="data_augmentation",
)

# Optimasi pipeline: prefetch agar training lebih cepat
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# ======================================================
# 2. ARSITEKTUR MODEL (VGG16 - FULL FREEZE)
# ======================================================
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Kunci seluruh convolutional layers (pure transfer learning)

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)  # Normalisasi khusus VGG16
x = base_model(x, training=False)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs, name="batik_aceh_vgg16")

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ======================================================
# 3. TRAINING
# ======================================================
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping],
)

# ======================================================
# 4. SIMPAN MODEL
# ======================================================
model.save(MODEL_OUTPUT_PATH)
print(f"Model berhasil disimpan ke: {MODEL_OUTPUT_PATH}")
print(f"Daftar kelas disimpan ke: {CLASS_NAMES_PATH}")

# ======================================================
# 5. EVALUASI MODEL
# ======================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

print("Mengevaluasi model pada data validasi...")
# Ambil label asli dan prediksi
y_true = []
y_pred_probs = []

# Matikan shuffle di validation dataset jika ada, tapi karena dari image_dataset_from_directory, 
# kita bisa iterasi langsung. Pastikan kita tidak memanggil .shuffle() sebelumnya.
for images, labels in validation_dataset:
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    preds = model.predict(images, verbose=0)
    y_pred_probs.extend(preds)

y_true = np.array(y_true)
y_pred = np.argmax(y_pred_probs, axis=1)

# Simpan history training
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Akurasi Model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
print("Grafik training disimpan ke: training_history.png")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Label Asli')
plt.xlabel('Label Prediksi')
plt.savefig('confusion_matrix.png')
print("Confusion matrix disimpan ke: confusion_matrix.png")

# Classification Report (F1-score, dll)
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
with open("metrics.json", "w") as f:
    json.dump(report, f, indent=4)
print("Metrik evaluasi disimpan ke: metrics.json")
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=class_names))

