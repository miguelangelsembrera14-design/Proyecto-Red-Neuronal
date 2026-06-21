"""Definición de la arquitectura CNN."""
import tensorflow as tf
from tensorflow.keras import layers, models


def construir_modelo(img_size, n_clases, learning_rate):
    modelo = models.Sequential([
        # Bloque 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(*img_size, 3)),
        layers.MaxPooling2D(2, 2),
        # Bloque 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        # Bloque 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        # Clasificador
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(n_clases, activation="softmax"),
    ], name="CNN_Clasificador")

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo