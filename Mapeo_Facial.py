import face_recognition
import cv2
import os
import numpy as np
from scipy.spatial import distance
import dlib
from imutils.video import VideoStream
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
import imutils

# Intalar en Bash
# pip install face_recognition opencv-python dlib imutils pillow smtplib
# pip install dlib face_recognition opencv-python keras tensorflow


# Carpeta con las imágenes de referencia
known_faces_dir = 'ruta/a/tu/carpeta/de/caras'
unknown_faces_dir = 'ruta/a/tu/carpeta/para/desconocidos'

# Crear la carpeta para caras desconocidas si no existe
os.makedirs(unknown_faces_dir, exist_ok=True)

# Configuración de correo electrónico (opcional)
SMTP_SERVER = 'smtp.example.com'
SMTP_PORT = 587
EMAIL_ADDRESS = 'tu_correo@example.com'
EMAIL_PASSWORD = 'tu_contraseña'
RECIPIENT_EMAIL = 'destinatario@example.com'

# Configuración de logging
logging.basicConfig(filename='reconocimiento_facial.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar el modelo para la detección de emociones (ejemplo con FER2013)
import keras.models as km
emotion_model = km.load_model('fer2013_mini_XCEPTION.119-0.65.hdf5')
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# Inicializar el detector de caras y la detección de puntos clave
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Cargar las imágenes conocidas y sus codificaciones
known_face_encodings = []
known_face_names = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)

# Inicializar el video stream
vs = VideoStream(src=0).start()
time.sleep(2.0)

def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    text = msg.as_string()
    server.sendmail(EMAIL_ADDRESS, RECIPIENT_EMAIL, text)
    server.quit()

def detect_emotion(face_image):
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    roi_gray = np.expand_dims(np.expand_dims(roi_gray, -1), 0)
    predictions = emotion_model.predict(roi_gray)[0]
    max_index = np.argmax(predictions)
    return EMOTIONS[max_index], predictions

# Bucle principal
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            logging.info(f"Reconocido: {name}")

        # Dibujar el cuadro alrededor de la cara
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)

        # Obtener los puntos clave de la cara para el seguimiento facial
        dlib_rect = dlib.rectangle(left, top, right, bottom)
        shape = predictor(gray, dlib_rect)
        landmarks = np.array([(p.x, p.y) for p in shape.parts()])

        # Dibujar los puntos clave (opcional)
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

        # Detectar la emoción en la cara
        face_image = frame[top:bottom, left:right]
        emotion, probabilities = detect_emotion(face_image)
        text = f"{name} ({emotion})"
        cv2.putText(frame, text, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)

        # Enviar correo electrónico si la persona es desconocida
        if name == "Unknown":
            unknown_face_path = os.path.join(unknown_faces_dir, f"unknown_{time.time()}.jpg")
            cv2.imwrite(unknown_face_path, face_image)
            send_email("Nueva cara detectada", f"Nueva cara detectada y guardada en {unknown_face_path}")
            logging.info(f"Desconocido: Guardado en {unknown_face_path}")

    # Mostrar el frame resultante
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()


