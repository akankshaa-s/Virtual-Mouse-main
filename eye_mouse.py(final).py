import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import os
import glob
import time

# Load Haar cascades for face and eye detection
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'

# Check if the cascade files exist
if not os.path.exists(face_cascade_path):
    print("Face cascade file not found!")
if not os.path.exists(eye_cascade_path):
    print("Eye cascade file not found!")

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# Path to save eye ROIs
eye_dataset_path = 'C:/Users/sse/Virtual-Mouse-main/eye_dataset'
os.makedirs(eye_dataset_path, exist_ok=True)

# Get screen dimensions
screen_width, screen_height = pyautogui.size()  # Define screen dimensions

# VideoWriter to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
output_path = 'C:/Users/sse/Virtual-Mouse-main/output_video.mp4'
frame_rate = 30.0  # You can adjust the frame rate

# Initialize VideoWriter object
out = cv2.VideoWriter(output_path, fourcc, frame_rate, (screen_width, screen_height))

if not out.isOpened():
    print("Failed to open VideoWriter.")
    exit()

print(f"VideoWriter initialized: {output_path}, Frame size: {(screen_width, screen_height)}, Frame rate: {frame_rate}")

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Function to log eye distances
def log_eye_distances(eye_distance):
    with open('eye_distances.log', 'a') as f:
        f.write(f"{time.time()},{eye_distance}\n")

# Initialize VideoCapture for the camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Failed to open camera.")
    exit()

while True:  # Run indefinitely
    ret, frame = cam.read()  # Read a frame from the camera
    if not ret:
        print("Failed to read frame from camera.")
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    frame_height, frame_width, _ = frame.shape  # Get frame dimensions

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    output = face_mesh.process(rgb_frame)  # Process the RGB frame for face landmarks

    landmark_points = output.multi_face_landmarks

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

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

        # Print debug information for landmark positions
        print(f"Left eye top: {left_eye_top.y}, bottom: {left_eye_bottom.y}")

        for landmark in [left_eye_top, left_eye_bottom]:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        eye_distance = left_eye_top.y - left_eye_bottom.y
        print(f"Eye distance: {eye_distance}")  # Debugging output
        log_eye_distances(eye_distance)  # Log eye distance

        if eye_distance < 0.01:  # Adjust the threshold if needed
            pyautogui.click()
            pyautogui.sleep(0.5)

    if len(faces) == 0:
        print("No faces detected.")
        time.sleep(0.1)  # Prevent rapid looping when no faces are detected
        continue

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]  # Use gray scale for eye detection
        eyes = eye_cascade.detectMultiScale(face_roi)

        print(f"Faces detected: {len(faces)}, Eyes detected: {len(eyes)}")  # Debug output

        for (ex, ey, ew, eh) in eyes:
            eye_roi = face_roi[ey:ey + eh, ex:ex + ew]
            label = "eye"
            eye_filename = os.path.join(eye_dataset_path, f"{label}{len(glob.glob(os.path.join(eye_dataset_path, f'{label}*.jpg'))) + 1}.jpg")

            print(f"Saving eye ROI to: {eye_dataset_path}")  # Debug output

            try:
                cv2.imwrite(eye_filename, eye_roi)
                if os.path.exists(eye_filename):
                    print(f"Successfully saved {eye_filename}")  # Debug output
                else:
                    print(f"Failed to save {eye_filename}")  # Debug output
            except Exception as e:
                print(f"Error saving eye ROI: {e}")

            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

    # Capture the screen
    screenshot = pyautogui.screenshot()
    screen_frame = np.array(screenshot)
    screen_frame = cv2.cvtColor(screen_frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    # Add the camera frame as a small picture-in-picture in the corner of the screen frame
    camera_frame_small = cv2.resize(frame, (320, 240))  # Resize camera frame to a smaller size
    screen_frame[10:250, 10:330] = camera_frame_small  # Position the small frame on the top-left corner

    # Write the screen frame to the video file
    out.write(screen_frame)

    # Display the screen frame
    cv2.imshow("Screen Recording", screen_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the video capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved to {output_path}")