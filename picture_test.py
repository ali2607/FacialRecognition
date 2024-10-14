import cv2
import numpy as np
from keras._tf_keras.keras.models import load_model

# Load the pre-trained emotion detection model
EmotionModel = load_model('emotion_detection.keras')

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define the labels for the emotion classes
labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

# Read the image from the specified file
frame = cv2.imread('FacesImages\\test.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray, minNeighbors=5)

# Iterate over each detected face
for (x, y, w, h) in faces:
    # Extract the face region of interest (ROI)
    face = gray[y:y + h, x:x + w]

    # Resize the face ROI to 48x48 pixels (input size expected by the model)
    resizedface = cv2.resize(face, (48, 48))

    # Reshape the face ROI to match the input shape of the model
    reshapedface = np.reshape(resizedface, (1, 48, 48, 1))

    # Predict the emotion probabilities for the face ROI
    emotionResult = EmotionModel.predict(reshapedface)

    # Get the index of the emotion with the highest probability
    label = np.argmax(emotionResult)

    # Draw a rectangle around the face in the original image
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Put the emotion label text above the face rectangle
    cv2.putText(
        frame,
        labels[label],
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

# Display the image with emotion detection
cv2.imshow('Emotion Detection', frame)

# Wait indefinitely until a key is pressed
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()
