import cv2
import numpy as np
from keras._tf_keras.keras.models import load_model

EmotionModel = load_model('emotion_detection.keras')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels = {0:'Angry', 1:'Disgust',2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, minNeighbors= 5)
    for x, y, w, h in faces:
        face = gray[y:y+h, x:x+w]
        resizedface = cv2.resize(face, (48, 48))
        reshapedface = np.reshape(resizedface, (1, 48, 48, 1))
        emotionResult = EmotionModel.predict(reshapedface)
        label = np.argmax(emotionResult)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, labels[label], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detection', frame)
    key = cv2.waitKey(1)
    if key == ord('x'):
        break