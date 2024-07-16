import os
import sys
import cv2
import mediapipe as mp
import pyautogui
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Set the KERAS_HOME environment variable to avoid UnicodeEncodeError
os.environ['KERAS_HOME'] = 'G:/Virtual-Mouse-main/keras/'
# Set the PYTHONIOENCODING environment variable to avoid UnicodeEncodeError
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Redirect stdout to a UTF-8 encoded stream
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# Define directories
train_dir = 'G:/Virtual-Mouse-main/datasets/train/'
validation_dir = 'G:/Virtual-Mouse-main/datasets/validation/'
img_height, img_width = 224, 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_data = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Debugging: Print shapes of data batches
x_batch, y_batch = next(train_data)
print(f"x_batch shape: {x_batch.shape}")
print(f"y_batch shape: {y_batch.shape}")

# Number of classes
num_classes = len(train_data.class_indices)

# Load a pretrained model (e.g., MobileNetV2) with weights trained on ImageNet
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base layers
base_model.trainable = False

# Add custom head for gesture classification
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

# Build the model
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
x = global_average_layer(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=validation_data
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_data)
print(f'Validation accuracy: {accuracy}')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize hand detection
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Get screen dimensions
screen_width, screen_height = pyautogui.size()
index_x = index_y = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Convert frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hand_detector.process(rgb_frame)

    # Process the frame to detect hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            # Extract index finger and thumb tip positions
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * frame_width), int(lm.y * frame_height)

                if id == 8:
                    index_x = int(lm.x * screen_width)
                    index_y = int(lm.y * screen_height)
                    pyautogui.moveTo(index_x, index_y)

                if id == 4:
                    thumb_x = int(lm.x * screen_width)
                    thumb_y = int(lm.y * screen_height)

                    # Check if index finger and thumb are close
                    if abs(index_x - thumb_x) < 30 and abs(index_y - thumb_y) < 30:
                        pyautogui.click()
                        pyautogui.sleep(0.5)

    # Display the resulting frame
    cv2.imshow('Hand Mouse', frame)

    # Exit loop on 'Esc' key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
