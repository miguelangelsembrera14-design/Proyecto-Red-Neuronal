import cv2
import numpy as np
import tensorflow as tf
import config

modelo = tf.keras.models.load_model(config.MODEL_PATH)
clases = np.load(config.CLASSES_PATH, allow_pickle=True)

cap = cv2.VideoCapture(0)  


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocesar el frame igual que las imágenes
    img = cv2.resize(frame, config.IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X = np.expand_dims(img.astype("float32") / 255.0, axis=0)

    # Predecir
    pred = modelo.predict(X, verbose=0)[0]
    idx = np.argmax(pred)
    clase = clases[idx]
    confianza = pred[idx]

    # Mostrar resultado en el video
    texto = f"{clase}: {confianza:.0%}"
    cv2.putText(frame, texto, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Mostrar todas las probabilidades al costado
    for i, (c, p) in enumerate(zip(clases, pred)):
        barra = f"{c}: {p:.0%}"
        cv2.putText(frame, barra, (10, 80 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Clasificador en tiempo real", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()