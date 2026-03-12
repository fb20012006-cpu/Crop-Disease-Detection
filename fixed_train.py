# ============================================================
# FIX - Correct paths found from your output
# PlantVillage is at /content/PVRaw/PlantVillage
# PlantDoc train is already in Drive
# Cassava is already in Drive
# Run this in a new cell
# ============================================================

import os, json, shutil, zipfile
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount("/content/drive")

DRIVE_BASE       = "/content/drive/MyDrive/CropDisease"
MODEL_SAVE_PATH  = DRIVE_BASE + "/crop_disease_model.h5"
CLASS_NAMES_PATH = DRIVE_BASE + "/class_names.json"
os.makedirs(DRIVE_BASE, exist_ok=True)

# ============================================================
# CORRECT PATHS
# ============================================================

# PlantVillage - two copies found, use the bigger one
PV_PATH1 = "/content/PVRaw/PlantVillage"          # 15 classes
PV_PATH2 = "/content/PVRaw/plantvillage/PlantVillage"  # may have more

# Pick the one with more classes
def count_img_classes(path):
    if not os.path.exists(path):
        return 0
    return len([c for c in os.listdir(path)
                if os.path.isdir(os.path.join(path, c)) and
                len([f for f in os.listdir(os.path.join(path, c))
                     if f.lower().endswith(('.jpg','.jpeg','.png'))]) >= 30])

c1 = count_img_classes(PV_PATH1)
c2 = count_img_classes(PV_PATH2)
print("PV_PATH1 classes: " + str(c1))
print("PV_PATH2 classes: " + str(c2))

PLANTVILLAGE_PATH = PV_PATH1 if c1 >= c2 else PV_PATH2
print("Using: " + PLANTVILLAGE_PATH)

# PlantDoc - saved to Drive by previous run
PLANTDOC_PATH = DRIVE_BASE + "/PlantDoc/train"

# Cassava - saved to Drive by previous run
CASSAVA_PATH = DRIVE_BASE + "/CassavaDisease"

# ============================================================
# SHOW WHAT WE HAVE
# ============================================================
print("")
print("=== Dataset Summary ===")

def show_dataset(name, path):
    if not os.path.exists(path):
        print(name + ": NOT FOUND at " + path)
        return []
    classes = [c for c in os.listdir(path)
               if os.path.isdir(os.path.join(path, c))]
    total_imgs = sum(
        len([f for f in os.listdir(os.path.join(path, c))
             if f.lower().endswith(('.jpg','.jpeg','.png'))])
        for c in classes
    )
    print(name + ": " + str(len(classes)) + " classes, " + str(total_imgs) + " images")
    return classes

pv_classes  = show_dataset("PlantVillage", PLANTVILLAGE_PATH)
pd_classes  = show_dataset("PlantDoc",     PLANTDOC_PATH)
cas_classes = show_dataset("Cassava",      CASSAVA_PATH)

# ============================================================
# BUILD IMAGE LIST (no file copying - fast!)
# ============================================================
print("")
print("=== Building combined image list ===")

MIN_IMAGES = 30
all_paths  = []
all_labels = []
class_names = []
seen = set()

def collect(dataset_path, prefix=""):
    if not os.path.exists(dataset_path):
        return 0
    added = 0
    for cls in sorted(os.listdir(dataset_path)):
        cls_path = os.path.join(dataset_path, cls)
        if not os.path.isdir(cls_path):
            continue
        imgs = [f for f in os.listdir(cls_path)
                if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if len(imgs) < MIN_IMAGES:
            continue
        name = prefix + cls if prefix else cls
        if name in seen:
            continue
        seen.add(name)
        class_names.append(name)
        added += 1
    return added

a = collect(PLANTVILLAGE_PATH)
print("PlantVillage: " + str(a) + " classes")
a = collect(PLANTDOC_PATH, prefix="doc_")
print("PlantDoc:     " + str(a) + " classes")
a = collect(CASSAVA_PATH, prefix="cas_")
print("Cassava:      " + str(a) + " classes")

class_names = sorted(class_names)
NUM_CLASSES = len(class_names)
print("")
print("Total classes: " + str(NUM_CLASSES))
print("All classes:")
for c in class_names:
    print("  " + c)

# Now build path+label lists
class_to_idx = {c: i for i, c in enumerate(class_names)}

def add_images(dataset_path, prefix=""):
    if not os.path.exists(dataset_path):
        return 0
    count = 0
    for cls in os.listdir(dataset_path):
        cls_path = os.path.join(dataset_path, cls)
        if not os.path.isdir(cls_path):
            continue
        name = prefix + cls if prefix else cls
        if name not in class_to_idx:
            continue
        idx = class_to_idx[name]
        for img in os.listdir(cls_path):
            if img.lower().endswith(('.jpg','.jpeg','.png')):
                all_paths.append(os.path.join(cls_path, img))
                all_labels.append(idx)
                count += 1
    return count

n = add_images(PLANTVILLAGE_PATH)
print("PlantVillage images: " + str(n))
n = add_images(PLANTDOC_PATH, prefix="doc_")
print("PlantDoc images:     " + str(n))
n = add_images(CASSAVA_PATH, prefix="cas_")
print("Cassava images:      " + str(n))

print("")
print("TOTAL: " + str(NUM_CLASSES) + " classes, " + str(len(all_paths)) + " images")

with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=2)
print("Class names saved to Drive")

# ============================================================
# DATA PIPELINE
# ============================================================
print("")
print("=== Building pipeline ===")

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    all_paths, all_labels, test_size=0.30, stratify=all_labels, random_state=42)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.33, stratify=temp_labels, random_state=42)

print("Train: " + str(len(train_paths)) + " | Val: " + str(len(val_paths)) + " | Test: " + str(len(test_paths)))

aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.1),
    layers.RandomContrast(0.1),
])

def load_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def load_augment(path, label):
    img, label = load_preprocess(path, label)
    img = aug(img, training=True)
    return img, label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = (tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
            .shuffle(len(train_paths))
            .map(load_augment, num_parallel_calls=AUTOTUNE)
            .batch(32).prefetch(AUTOTUNE))
val_ds   = (tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
            .map(load_preprocess, num_parallel_calls=AUTOTUNE)
            .batch(32).prefetch(AUTOTUNE))
test_ds  = (tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
            .map(load_preprocess, num_parallel_calls=AUTOTUNE)
            .batch(32).prefetch(AUTOTUNE))

print("Pipeline ready!")

# ============================================================
# BUILD + TRAIN
# ============================================================
print("")
print("=== Training ===")

base    = MobileNetV2(input_shape=(224,224,3), include_top=False, weights="imagenet")
base.trainable = False
inputs  = layers.Input(shape=(224,224,3))
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(512, activation="relu")(x)
x       = layers.Dropout(0.4)(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model   = Model(inputs, outputs)

# Phase 1 - head only
print("Phase 1: Head training...")
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h1 = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[
    EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
])
print("Phase 1 best: " + str(round(max(h1.history["val_accuracy"]) * 100, 2)) + "%")

# Phase 2 - fine tune
print("Phase 2: Fine-tuning...")
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h2 = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
])
print("Phase 2 best: " + str(round(max(h2.history["val_accuracy"]) * 100, 2)) + "%")

# Evaluate
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print("")
print("Test Accuracy: " + str(round(test_acc * 100, 2)) + "%")
model.save(MODEL_SAVE_PATH)
print("Model saved to Drive!")

# Download
try:
    from google.colab import files
    files.download(MODEL_SAVE_PATH)
    files.download(CLASS_NAMES_PATH)
    print("Downloading to laptop!")
except Exception:
    print("Get from Drive: " + DRIVE_BASE)

# Print for app.py
print("")
print("CLASS_NAMES = [")
for name in class_names:
    print("    '" + name + "',")
print("]")
