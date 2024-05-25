import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

# Load the trained model
model = load_model('face_shape_recognition_model.h5')

# Define the face shape categories
face_shape_categories = ['heart', 'round', 'oblong', 'square', 'oval']

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to preprocess the image
def preprocess_image(image):
    image_resized = cv2.resize(image, (128, 128))  # Resize to match the input shape of the model
    image_normalized = image_resized / 255.0  # Normalize pixel values to [0, 1]
    image_expanded = np.expand_dims(image_normalized, axis=0)  # Expand dimensions to match model input
    return image_expanded

# Function to detect faces in an image
def detect_faces(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Function to capture image from camera and predict face shape
def capture_and_predict():
    cap = cv2.VideoCapture(0)  # Open the camera

    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow('Press "q" to capture', frame)  # Display the frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows

    if frame is not None:
        faces = detect_faces(frame)  # Detect faces in the captured image
        if len(faces) == 0:
            print("No face detected.")
            return

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]  # Extract the face region
            preprocessed_face = preprocess_image(face)  # Preprocess the face region
            prediction = model.predict(preprocessed_face)  # Predict the face shape
            predicted_label = face_shape_categories[np.argmax(prediction)]  # Get the predicted label

            # Draw a rectangle around the face and label it with the predicted face shape
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the final image with predictions
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        label.config(image=img_tk)
        label.image = img_tk
    else:
        print("No frame captured.")

# Create the main window
root = tk.Tk()
root.title("Face Shape Recognition")
root.geometry("800x600")

# Create and place a label for displaying the image
label = Label(root)
label.pack()

# Create and place a button to capture the image
capture_button = Button(root, text="Capture Image and Predict Face Shape", command=capture_and_predict)
capture_button.pack()

# Start the GUI event loop
root.mainloop()
