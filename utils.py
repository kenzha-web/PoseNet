"""
Utility functions for rendering annotated images using Mediapipe Holistic for different views.

1. Top View Landmark Image:
   - Converts BGR frame to RGB for Mediapipe processing.
   - Processes the frame using the Holistic model to detect and annotate landmarks for hands and body pose.
   - Converts the annotated RGB image back to BGR format for display.
   - Draws landmarks for right and left hands with specific color and thickness.
   - Draws landmarks for full body pose with specific color and thickness.
   - Returns the annotated BGR image and results object.

2. Front View Landmark Image:
   - Converts BGR frame to RGB for Mediapipe processing.
   - Processes the frame using the Holistic model to detect and annotate landmarks for facial landmarks.
   - Converts the annotated RGB image back to BGR format for display.
   - Draws facial landmarks with specific color and thickness.
   - Returns the annotated BGR image and results object.
"""

import cv2

def top_view_lendmark_image(frame, holistic, mp_holistic, mp_drawing):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                              )

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                              )

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                              )

    return (image, results)

def front_view_lendmark_image(frame, holistic, mp_holistic, mp_drawing):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )

    return (image, results)