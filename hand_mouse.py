import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)  # capture 1st video source
hand_detector = mp.solutions.hands.Hands()  # detect hands
drawing_utils = mp.solutions.drawing_utils  # util to draw landmarks on hand
screen_width, screen_height = pyautogui.size()  # get screen dimensions
index_x = index_y = 0
scrolling = False

while True:  # run forever
    _, frame = cap.read()  # read whatever is captured by cam
    frame = cv2.flip(frame, 1)  # flip frame vertically
    frame_height, frame_width, _ = frame.shape  # get frame dimensions

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # change color of frame to RGB
    output = hand_detector.process(rgb_frame)  # process rgb_frame and store in output

    hands = output.multi_hand_landmarks  # get landmarks of hands from output

    if hands:  # if hand exists on screen
        for hand in hands:  # for every hand
            drawing_utils.draw_landmarks(frame, hand)  # draw the points on the frame
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):  # iterate through landmarks
                x = int(landmark.x * frame_width)  # get position of hand in width
                y = int(landmark.y * frame_height)  # get position of hand in height

                if id == 8:  # select index finger tip
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)  # draw circle around index finger tip
                    index_x = landmark.x * screen_width  # get position of cursor along width
                    index_y = landmark.y * screen_height  # get position of cursor along height
                    pyautogui.moveTo(index_x, index_y)  # move cursor to given location

                if id == 4:  # select thumb finger tip
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)  # draw circle around thumb finger tip
                    thumb_x = landmark.x * screen_width  # get position of thumb along width
                    thumb_y = landmark.y * screen_height  # get position of thumb along height

                    if abs(index_y - thumb_y) < 30 and abs(index_x - thumb_x) < 30:  # if both points touch
                        pyautogui.click()  # click
                        pyautogui.sleep(0.5)  # wait for 0.5 sec

                if id == 12:  # select middle finger tip for scrolling
                    middle_x = landmark.x * screen_width
                    middle_y = landmark.y * screen_height
                    if not scrolling and abs(index_y - middle_y) < 30 and abs(index_x - middle_x) < 30:
                        scrolling = True
                        initial_y = middle_y
                    elif scrolling:
                        pyautogui.scroll((initial_y - middle_y) / 5)
                        scrolling = False

    cv2.imshow('Hand Mouse', frame)  # show frame with the window name 'Hand Mouse'
    cv2.waitKey(1)  # wait for 1 ms

    cap.release()
    cv2.destroyAllWindows()