"""
Script for capturing movement coordinates using webcam and writing them to a CSV file.

1. Setup:
   - Initializes necessary libraries and utilities (OpenCV, Mediapipe).
   - Defines paths and settings for CSV file handling.

2. Coordinate Writing Function:
   - Defines 'write_results' function to extract pose landmarks from webcam frames.
   - Writes landmarks' coordinates along with a specified class label to 'movement_classes_coordinates.csv'.

3. Webcam Capture and Processing Loop:
   - Initializes webcam capture with OpenCV.
   - Starts a loop to continuously capture frames and process them for movement landmarks.
   - Invokes 'write_results' to append landmark coordinates to the CSV file.
   - Displays processed frames with annotated landmarks using OpenCV imshow.
   - Ends capture after 20 seconds or upon user pressing 'q' key.
"""

import csv
import os
import numpy as np
import mediapipe as mp
import cv2
import time
from utils import top_view_lendmark_image

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

csv_file_path = 'movement_classes_coordinates.csv'
csv_file_exists = os.path.exists(csv_file_path)
csv_file_mode = 'a' if csv_file_exists else 'w'

cap = cv2.VideoCapture(1)

def write_results(results):
    """
        Extracts pose landmarks from results and writes them along with a class label to a CSV file.

        Args:
        - results: Mediapipe Holistic results containing pose landmarks.

        Writes:
        - Appends each pose landmark's coordinates and associated class label to 'movement_classes_coordinates.csv'.
    """

    if results.pose_landmarks:
        num_pose_coords = len(
            results.pose_landmarks.landmark)
        num_total_coords = num_pose_coords

        landmarks = ['class']
        for val in range(1, num_total_coords + 1):
            landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

        if not csv_file_exists:
            with open(csv_file_path, mode='w', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(landmarks)

        # class_name = "Normal condition"
        # class_name = "The left-hand is lowered down"
        # class_name = "The right-hand is lowered down"
        class_name = "Arm position is down"
        # class_name = "Hands and fingers are not visible"
        # class_name = "Head position turned right"
        # class_name = "Head position turned left"
        # class_name = "Calling the support"

        # if results.right_hand_landmarks or results.left_hand_landmarks:
        #     class_name = "Normal condition"
        # else:
        #     class_name = "Hands and fingers are not visible"

        try:
            pose = results.pose_landmarks.landmark
            pose_row = list(
                np.array(
                    [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            row = pose_row
            row.insert(0, class_name)

            with open(csv_file_path, mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)

        except:
            pass

# Initialize Mediapipe Holistic model for pose estimation
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.8) as holistic:
    start_time = time.time()

    while cap.isOpened():
        _, frame = cap.read()
        image, res1 = top_view_lendmark_image(frame, holistic, mp_holistic, mp_drawing)
        write_results(res1)
        cv2.imshow('Webcam test', image)
        current_time = time.time()

        if current_time - start_time >= 20:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
