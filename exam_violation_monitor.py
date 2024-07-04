"""
Script for analyzing pre-recorded video footage to classify body language using a pretrained ML model.

1. Setup:
   - Loads necessary libraries (OpenCV, Mediapipe) and pretrained model from 'movement_classification_data.pkl'.
   - Defines paths and settings for CSV and Excel file handling.

2. Data Initialization:
   - Initializes an empty DataFrame 'data' to store classified body language and timestamps.
   - Checks for existing CSV file 'movement_classes_coordinates.csv' for data appending.

3. Functions:
   - 'append_data_to_excel': Appends DataFrame 'data' to 'violation_journal.xlsx'.
   - 'create_lines': Draws classification results on the video frame.
   - 'fill_dataframe': Updates DataFrame 'data' with new classification if different from the last recorded.

4. Video Processing Loop:
   - Reads frames from 'Exam.mp4' and processes them using Mediapipe Holistic model.
   - Classifies body language using the pretrained model and updates 'data'.
   - Displays annotated video frames with classification results using OpenCV imshow.
   - Ends processing after the video ends or upon user pressing 'q' key.
"""

import os
import numpy as np
import mediapipe as mp
import cv2
import pickle
import pandas as pd
import time
from openpyxl import load_workbook

from utils import top_view_lendmark_image

with open('movement_classification_data.pkl', 'rb') as f:
    model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

excel_file_path = 'violation_journal.xlsx'
data = pd.DataFrame(columns=['Class', 'Time'])

csv_file_path = 'movement_classes_coordinates.csv'
csv_file_exists = os.path.exists(csv_file_path)
csv_file_mode = 'a' if csv_file_exists else 'w'

video_file_path = 'Exam.mp4'
cap = cv2.VideoCapture(video_file_path)

def append_data_to_excel(file_path, df):
    if not os.path.exists(file_path):
        df.to_excel(file_path, index=False)
    else:
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)


def create_lines(coords: tuple, body_language_class, body_language_prob):
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


def fill_dataframe(body_language_class):
    global data

    last_class, last_time = None, None
    time_in_seconds = frame_number / fps
    minutes = int(time_in_seconds // 60)
    seconds = int(time_in_seconds % 60)

    if not data.empty:
        last_row_in_dataframe = data.iloc[-1:]
        last_class = last_row_in_dataframe.Class.values[0]
        last_time = last_row_in_dataframe.Time.values[0]

    if body_language_class != last_class and last_time != f"{minutes}:{seconds:02d}":
        data = pd.concat(
            [data,
             pd.DataFrame({"Class": [body_language_class],
                           "Time": [f"{minutes}:{seconds:02d}"]
                           }
                          )
             ],
            ignore_index=True
        )
    print(f"Записан новый класс: {body_language_class} в {minutes}:{seconds:02d}")


def draw_lines(results, image, frame_number, fps):
    global data

    try:
        pose = results.pose_landmarks.landmark
        pose_row = list(
            np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten()
        )
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

        create_lines(coords, body_language_class, body_language_prob)

        if body_language_class and body_language_class != "Normal condition":
            fill_dataframe(body_language_class)

    except Exception as e:
        print(f"Error: {e}")


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.8) as holistic:
    frame_number = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_time = time.time()

    while cap.isOpened():
        ret1, frame = cap.read()

        if not ret1:
            break

        frame_number += 1
        image, res1 = top_view_lendmark_image(frame, holistic, mp_holistic, mp_drawing)
        draw_lines(res1, image, frame_number, fps)
        cv2.imshow('Webcam test', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    append_data_to_excel(excel_file_path, data)

cap.release()
cv2.destroyAllWindows()
print(f"Файл Excel сохранен: {excel_file_path}")
