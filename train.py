"""
Script principal de entrenamiento del clasificador CNN.

Uso:
    python train.py
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks

import config
from dataset import preparar_datos
from modelo import construir_modelo


def main():
    X_train, X_val, X_test, y_train, y_val, y_test, le = preparar_datos(
        config.DATASET_DIR, config.IMG_SIZE
    )

    n_clases = len(le.classes_)
    modelo = construir_modelo(config.IMG_SIZE, n_clases, config.LEARNING_RATE)
    modelo.summary()

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
        ),
        callbacks.ModelCheckpoint(
            config.MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
    ]

    history = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=cb_list,
    )

    # Evaluación en test
    loss, acc = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"Loss en test     : {loss:.4f}")
    print(f"Accuracy en test : {acc:.2%}")

    # Gráficas
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Métricas de Entrenamiento — CNN", fontsize=14, fontweight="bold")

    axes[0].plot(history.history["accuracy"], label="Train", color="#2E75B6", lw=2)
    axes[0].plot(history.history["val_accuracy"], label="Validación", color="#E07B39", lw=2, ls="--")
    axes[0].set_title("Accuracy por época")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train", color="#2E75B6", lw=2)
    axes[1].plot(history.history["val_loss"], label="Validación", color="#E07B39", lw=2, ls="--")
    axes[1].set_title("Loss por época")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.CURVES_PATH, dpi=150)
    plt.show()

    # Guardar etiquetas (el modelo ya se guarda solo vía ModelCheckpoint)
    np.save(config.CLASSES_PATH, le.classes_)
    print("✅ Modelo, clases y gráficas guardados localmente")


if __name__ == "__main__":
    main()