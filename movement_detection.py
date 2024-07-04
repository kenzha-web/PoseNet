"""
Script for real-time inference of movement classification using a pre-trained machine learning model.

1. Model Loading:
   - Loads the trained Random Forest model from 'movement_classification_data.pkl' using pickle.

2. Setup:
   - Initializes necessary libraries and utilities (OpenCV, Mediapipe).
   - Defines paths and settings for CSV file handling.

3. Drawing Function:
   - Defines 'draw_lines' function to extract pose landmarks from webcam frames,
     classify the movement using the loaded model, and annotate the frame with classification results.

4. Webcam Capture and Processing Loop:
   - Initializes webcam capture with OpenCV.
   - Starts a loop to continuously capture frames and process them for movement classification.
   - Invokes 'draw_lines' to classify movements and visualize results on the webcam feed using OpenCV imshow.
   - Ends capture upon user pressing 'q' key.
"""

import os
import numpy as np
import mediapipe as mp
import cv2
import pickle
import pandas as pd

from utils import top_view_lendmark_image

# Load pre-trained Random Forest model for movement classification
with open('movement_classification_data.pkl', 'rb') as f:
    model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Path to store movement coordinates CSV file (not used for inference)
csv_file_path = 'movement_classes_coordinates.csv'
csv_file_exists = os.path.exists(csv_file_path)
csv_file_mode = 'a' if csv_file_exists else 'w'

# Initialize webcam capture
cap = cv2.VideoCapture(1)

def draw_lines(results, image):
    """
        Processes pose landmarks from Mediapipe Holistic results, classifies the movement using
        a pre-trained model, and annotates the image with classification results.

        Args:
        - results: Mediapipe Holistic results containing pose landmarks.
        - image: Frame from webcam capture to annotate with movement classification.

        Draws:
        - Annotated image with predicted movement class, class-specific text, and probability.

    """

    try:
        pose = results.pose_landmarks.landmark
        pose_row = list(
            np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

        row = pose_row

        X = pd.DataFrame([row])
        body_language_class = model.predict(X)[0]
        body_language_prob = model.predict_proba(X)[0]
        print(body_language_class, body_language_prob)

        coords = tuple(np.multiply(
            np.array(
                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
            , [640, 480]).astype(int))

        cv2.rectangle(image,
                      (coords[0], coords[1] + 10),
                      (coords[0] + len(body_language_class) * 40, coords[1] - 50),
                      (0, 0, 255), -1)

        cv2.putText(image, body_language_class, coords,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.rectangle(image, (0, 0), (250, 250), (0, 0, 255), -1)

        cv2.putText(image, 'Class'
                    , (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, body_language_class.split(' ')[0]
                    , (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, 'Probability'
                    , (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                    , (15, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    except:
        pass

# Initialize Mediapipe Holistic model for pose estimation
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.8) as holistic:
    while cap.isOpened():
        ret1, frame = cap.read()
        image, res1 = top_view_lendmark_image(frame, holistic, mp_holistic, mp_drawing)
        draw_lines(res1, image)
        cv2.imshow('Webcam test', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()