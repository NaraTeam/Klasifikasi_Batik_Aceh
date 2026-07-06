import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import json
import os

DATASET_DIR = r"E:\Batik3dataset\dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = "batik_aceh_vgg16.keras"
CENTROIDS_PATH = "centroids.npy"

print("Memuat dataset training...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False # Jangan shuffle agar labelnya berurutan, mudah dihitung
)

print("Memuat model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Ambil model sampai layer Dense(256) sebelum Dropout
# layer index: 
# -1: Dense (3)
# -2: Dropout
# -3: Dense (256)
feature_extractor = Model(inputs=model.input, outputs=model.layers[-3].output)

print("Mengekstrak fitur dari seluruh data training...")
features = []
labels_list = []

for images, labels in train_dataset:
    # Karena model kita inputnya memiliki data_augmentation bawaan di dalam arsitektur,
    # kita harus pastikan output fitur tidak teraugmentasi secara random saat inference
    feats = feature_extractor.predict(images, verbose=0)
    features.extend(feats)
    labels_list.extend(np.argmax(labels.numpy(), axis=1))

features = np.array(features)
labels_list = np.array(labels_list)

print("Menghitung centroid (rata-rata) tiap kelas...")
centroids = []
for i in range(3): # 3 classes
    class_features = features[labels_list == i]
    centroid = np.mean(class_features, axis=0)
    # Normalize centroid for cosine similarity
    centroid = centroid / np.linalg.norm(centroid)
    centroids.append(centroid)

centroids = np.array(centroids)
np.save(CENTROIDS_PATH, centroids)
print(f"Centroids disimpan ke {CENTROIDS_PATH} dengan shape {centroids.shape}")
