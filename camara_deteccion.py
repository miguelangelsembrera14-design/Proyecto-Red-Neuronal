import cv2
import numpy as np
import tensorflow as tf
import config

modelo = tf.keras.models.load_model(config.MODEL_PATH)
clases = np.load(config.CLASSES_PATH, allow_pickle=True)

COLORES = {
    "gato":  (0, 255, 100),
    "perro": (0, 180, 255),
    "auto":  (0, 0, 255),
    "bici":  (255, 200, 0),
}
COLOR_DEFAULT = (200, 200, 200)

def get_color(clase):
    for key in COLORES:
        if key in clase.lower():
            return COLORES[key]
    return COLOR_DEFAULT

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    margen_x = int(w * 0.20)
    margen_y = int(h * 0.20)
    x1, y1 = margen_x, margen_y
    x2, y2 = w - margen_x, h - margen_y

    region = frame[y1:y2, x1:x2]
    img = cv2.resize(region, config.IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X = np.expand_dims(img.astype("float32") / 255.0, axis=0)

    pred = modelo.predict(X, verbose=0)[0]
    idx = np.argmax(pred)
    clase = clases[idx]
    confianza = pred[idx]
    color = get_color(clase)

    # Oscurecer todo excepto la región central
    mascara = np.zeros_like(frame, dtype=np.uint8)
    mascara[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    frame_oscuro = (frame * 0.3).astype(np.uint8)
    frame_oscuro[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

    # Esquinas del cuadro (estilo moderno, no cuadro completo)
    largo = 40
    grosor = 3
    # Esquina superior izquierda
    cv2.line(frame_oscuro, (x1, y1), (x1 + largo, y1), color, grosor)
    cv2.line(frame_oscuro, (x1, y1), (x1, y1 + largo), color, grosor)
    # Esquina superior derecha
    cv2.line(frame_oscuro, (x2, y1), (x2 - largo, y1), color, grosor)
    cv2.line(frame_oscuro, (x2, y1), (x2, y1 + largo), color, grosor)
    # Esquina inferior izquierda
    cv2.line(frame_oscuro, (x1, y2), (x1 + largo, y2), color, grosor)
    cv2.line(frame_oscuro, (x1, y2), (x1, y2 - largo), color, grosor)
    # Esquina inferior derecha
    cv2.line(frame_oscuro, (x2, y2), (x2 - largo, y2), color, grosor)
    cv2.line(frame_oscuro, (x2, y2), (x2, y2 - largo), color, grosor)

    # Etiqueta principal centrada abajo del cuadro
    texto = f"{clase.upper()}  {confianza:.0%}"
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    tx = (w - tw) // 2
    ty = y2 + 40

    # Fondo de la etiqueta
    cv2.rectangle(frame_oscuro,
        (tx - 10, ty - th - 8), (tx + tw + 10, ty + 5),
        color, -1)
    cv2.putText(frame_oscuro, texto, (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # Barra de confianza abajo
    barra_w = int((w * 0.6) * confianza)
    barra_x = int(w * 0.20)
    barra_y = ty + 25
    cv2.rectangle(frame_oscuro,
        (barra_x, barra_y), (barra_x + int(w * 0.6), barra_y + 8),
        (60, 60, 60), -1)
    cv2.rectangle(frame_oscuro,
        (barra_x, barra_y), (barra_x + barra_w, barra_y + 8),
        color, -1)

    cv2.imshow("Mi CNN - Detector", frame_oscuro)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()