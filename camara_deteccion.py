import cv2
import numpy as np
import tensorflow as tf
import config

# ==========================
# Cargar modelo y clases
# ==========================
modelo = tf.keras.models.load_model(config.MODEL_PATH)
clases = np.load(config.CLASSES_PATH, allow_pickle=True)

# ==========================
# Colores (BGR)
# ==========================
COLORES = {
    "ave":            (0, 255, 255),    # Amarillo
    "botellaplas":    (255, 0, 255),    # Magenta
    "bus":            (0, 165, 255),    # Naranja
    "carro":          (0, 0, 255),      # Rojo
    "comedor":        (128, 0, 255),    # Morado
    "frutas":         (0, 255, 0),      # Verde
    "gato":           (0, 255, 100),    # Verde claro
    "lapices":        (255, 255, 0),    # Celeste
    "perro":          (255, 0, 0),      # Azul
    "personas":       (255, 255, 255),  # Blanco
    "refrigerador":   (180, 180, 180),  # Gris
    "reloj":          (255, 200, 0),    # Celeste claro
    "silla":          (255, 50, 150),   # Rosa
    "telefono":       (200, 50, 255),   # Violeta
    "television":     (50, 255, 200),   # Verde agua
}

COLOR_DEFAULT = (200, 200, 200)


def get_color(clase):
    for key in COLORES:
        if key.lower() == clase.lower():
            return COLORES[key]
    return COLOR_DEFAULT


# ==========================
# Abrir cámara
# ==========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Presiona Q o ESC para salir.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]

    # ==========================
    # Región central (60%)
    # ==========================
    margen_x = int(w * 0.20)
    margen_y = int(h * 0.20)

    x1 = margen_x
    y1 = margen_y
    x2 = w - margen_x
    y2 = h - margen_y

    region = frame[y1:y2, x1:x2]

    # ==========================
    # Preprocesamiento
    # ==========================
    img = cv2.resize(region, config.IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    X = np.expand_dims(
        img.astype("float32") / 255.0,
        axis=0
    )

    # ==========================
    # Predicción
    # ==========================
    pred = modelo.predict(X, verbose=0)[0]

    idx = np.argmax(pred)

    clase = clases[idx]

    confianza = pred[idx]

    color = get_color(clase)

    # ==========================
    # Fondo oscuro
    # ==========================
    frame_oscuro = (frame * 0.30).astype(np.uint8)

    frame_oscuro[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

    # ==========================
    # Esquinas futuristas
    # ==========================
    largo = 40
    grosor = 3

    # Superior izquierda
    cv2.line(frame_oscuro, (x1, y1), (x1 + largo, y1), color, grosor)
    cv2.line(frame_oscuro, (x1, y1), (x1, y1 + largo), color, grosor)

    # Superior derecha
    cv2.line(frame_oscuro, (x2, y1), (x2 - largo, y1), color, grosor)
    cv2.line(frame_oscuro, (x2, y1), (x2, y1 + largo), color, grosor)

    # Inferior izquierda
    cv2.line(frame_oscuro, (x1, y2), (x1 + largo, y2), color, grosor)
    cv2.line(frame_oscuro, (x1, y2), (x1, y2 - largo), color, grosor)

    # Inferior derecha
    cv2.line(frame_oscuro, (x2, y2), (x2 - largo, y2), color, grosor)
    cv2.line(frame_oscuro, (x2, y2), (x2, y2 - largo), color, grosor)

    # ==========================
    # Etiqueta
    # ==========================
    texto = f"{clase.upper()}   {confianza:.0%}"

    (tw, th), _ = cv2.getTextSize(
        texto,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        2
    )

    tx = (w - tw) // 2

    ty = y2 + 40

    cv2.rectangle(
        frame_oscuro,
        (tx - 10, ty - th - 10),
        (tx + tw + 10, ty + 5),
        color,
        -1,
    )

    cv2.putText(
        frame_oscuro,
        texto,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )

    # ==========================
    # Barra de confianza
    # ==========================
    ancho = int(w * 0.60)

    barra = int(ancho * confianza)

    bx = int(w * 0.20)

    by = ty + 25

    cv2.rectangle(
        frame_oscuro,
        (bx, by),
        (bx + ancho, by + 10),
        (60, 60, 60),
        -1,
    )

    cv2.rectangle(
        frame_oscuro,
        (bx, by),
        (bx + barra, by + 10),
        color,
        -1,
    )

    # ==========================
    # Mostrar
    # ==========================
    cv2.imshow("Mi CNN - Detector Inteligente", frame_oscuro)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord("q") or tecla == 27:
        break

# ==========================
# Liberar recursos
# ==========================
cap.release()
cv2.destroyAllWindows()