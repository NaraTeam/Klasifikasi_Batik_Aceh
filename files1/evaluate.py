import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

DATASET_DIR = r"E:\Batik3dataset\dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = "batik_aceh_vgg16.keras"
CLASS_NAMES_PATH = "class_names.json"

print("Memuat dataset validasi...")
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

print("Memuat model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("Membuat prediksi...")
y_true = []
y_pred_probs = []

for images, labels in validation_dataset:
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    preds = model.predict(images, verbose=0)
    y_pred_probs.extend(preds)

y_true = np.array(y_true)
y_pred = np.argmax(y_pred_probs, axis=1)

print("Menyimpan Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)
np.save('cm.npy', cm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Label Asli')
plt.xlabel('Label Prediksi')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

print("Menyimpan Metrics...")
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
with open("metrics.json", "w") as f:
    json.dump(report, f, indent=4)

print("Selesai! File confusion_matrix.png dan metrics.json berhasil dibuat.")
