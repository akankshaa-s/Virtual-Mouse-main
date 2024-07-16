import os
import logging

# Set environment variable to suppress INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Optionally, use the logging module to set the log level for TensorFlow
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf

from cProfile import label
from timeit import main
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import glob

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Path to save eye ROIs
eye_dataset_path = 'C:/Users/sse/Virtual-Mouse-main/eye_dataset'
os.makedirs(eye_dataset_path, exist_ok=True)

cam = cv2.VideoCapture(0)  # Capture video from the first video source
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_width, screen_height = pyautogui.size()  # Get screen dimensions

while True:
    # Capture screen
    img = pyautogui.screenshot()

    # Convert the screenshot to a numpy array
    frame = np.array(img)

    # Convert RGB to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    frame_height, frame_width, _ = frame.shape  # Get frame dimensions

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    results = face_mesh.process(rgb_frame)  # Process the RGB frame for face landmarks

    landmark_points = results.multi_face_landmarks

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if landmark_points:  # If face landmarks are detected
        landmarks = landmark_points[0].landmark  # Choose the first face

        for id, landmark in enumerate(landmarks[474:478]):  # Process specific eye landmarks
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            if id == 1:
                screen_x = landmark.x * screen_width
                screen_y = landmark.y * screen_height
                pyautogui.moveTo(screen_x, screen_y)

        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]

        for landmark in [left_eye_top, left_eye_bottom]:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        if left_eye_top.y - left_eye_bottom.y < 0.01:  # Adjust the threshold if needed
            pyautogui.click()
            pyautogui.sleep(0.5)

        eye_filename = os.path.join(eye_dataset_path,
                                    f"{label}{len(glob.glob(os.path.join(eye_dataset_path, f'{label}*.jpg'))) + 1}.jpg")

    # Check for 'q' key press to stop recording
    if cv2.waitKey(1) == ord('q'):
        break

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

cam.release()
cv2.destroyAllWindows()
