"""Carga y preprocesamiento del dataset de imágenes."""
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


def cargar_dataset(ruta_base, img_size):
    registros = []
    ruta_base = Path(ruta_base)

    for clase_dir in sorted(ruta_base.iterdir()):
        if not clase_dir.is_dir():
            continue
        label = clase_dir.name
        count = 0
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for img_path in clase_dir.glob(ext):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
                img = cv2.resize(img, img_size)
                registros.append({"path": str(img_path), "label": label, "pixels": img})
                count += 1
        print(f"  Clase '{label}': {count} imágenes")

    df = pd.DataFrame(registros)
    print(f"\n✅ Total: {len(df)} imágenes")
    return df


def preparar_datos(dataset_dir, img_size):
    """Carga el dataset y devuelve los splits train/val/test junto al LabelEncoder."""
    df = cargar_dataset(dataset_dir, img_size)

    X = np.stack(df["pixels"].values).astype("float32") / 255.0

    le = LabelEncoder()
    y_int = le.fit_transform(df["label"].values)
    y = tf.keras.utils.to_categorical(y_int)

    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")
    print(f"Clases  : {list(le.classes_)}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    print(f"\nTrain : {X_train.shape[0]} | Val : {X_val.shape[0]} | Test : {X_test.shape[0]}")

    return X_train, X_val, X_test, y_train, y_val, y_test, le