"""
Script de inferencia: clasifica una imagen como gato o perro.

Uso:
    python predict.py ruta/a/la/imagen.jpg
"""
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import config


def predecir(ruta_imagen: str):
    modelo = tf.keras.models.load_model(config.MODEL_PATH)
    clases = np.load(config.CLASSES_PATH, allow_pickle=True)

    img = cv2.imread(ruta_imagen)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta_imagen}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, config.IMG_SIZE)

    # Expandir a tensor (1, 64, 64, 3)
    X_pred = np.expand_dims(img_res.astype("float32") / 255.0, axis=0)

    pred = modelo.predict(X_pred, verbose=0)[0]
    idx = np.argmax(pred)
    clase = clases[idx]
    confianza = pred[idx]

    for c, p in zip(clases, pred):
        barra = "█" * int(p * 30)
        print(f"{c:<12} {barra} {p:.2%}")

    print(f"\n🎯 Predicción : {clase}  ({confianza:.2%} de confianza)")

    plt.figure(figsize=(5, 5))
    plt.imshow(img_rgb)
    plt.title(f"{clase}  —  {confianza:.2%}", fontsize=13, fontweight="bold")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clasifica una imagen como gato o perro")
    parser.add_argument("imagen", help="Ruta a la imagen a clasificar")
    args = parser.parse_args()
    predecir(args.imagen)