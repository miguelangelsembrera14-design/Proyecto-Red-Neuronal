"""
Descarga el dataset Microsoft Cats vs Dogs desde Kaggle y lo organiza
en dataset/gato/ y dataset/perro/.

Requiere tener configuradas las credenciales de Kaggle (una sola vez):
https://github.com/Kagglehub/kagglehub#authentication

Uso:
    python descargar_dataset.py
"""
import os
import shutil

import kagglehub

from config import DATASET_DIR


def descargar_y_organizar(n_por_clase: int = 500):
    path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset")
    print("Descargado en:", path)

    for clase, carpeta in [("gato", "Cat"), ("perro", "Dog")]:
        dst = os.path.join(DATASET_DIR, clase)
        os.makedirs(dst, exist_ok=True)
        src = os.path.join(path, "PetImages", carpeta)
        imgs = [f for f in os.listdir(src) if f.endswith(".jpg")][:n_por_clase]
        for img in imgs:
            shutil.copy(os.path.join(src, img), os.path.join(dst, img))

    for c in os.listdir(DATASET_DIR):
        n = len(os.listdir(os.path.join(DATASET_DIR, c)))
        print(f"  {c}: {n} imágenes")


if __name__ == "__main__":
    descargar_y_organizar()