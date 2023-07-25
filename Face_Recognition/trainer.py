import cv2
import os
import numpy as np

# Path to the directory containing the images of the person you want to recognize
path = "pictures"

# Create a face recognizer object
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Define a function to get the images and their corresponding labels
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    labels = []
    for image_path in image_paths:
        # Load the image and convert it to grayscale
        img = cv2.imread(image_path, 0)
        # Extract the label (the person's name) from the file name
        label = os.path.split(image_path)[-1].split(".")[0]
        # Detect the face in the image
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_rects = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
        # Add each face and its label to the faces and labels lists
        for (x,y,w,h) in face_rects:
            faces.append(img[y:y+h, x:x+w])
            labels.append(label)
    return faces, labels

# Get the images and labels for the person you want to recognize
faces, labels = get_images_and_labels(path)

labels = np.array(labels, dtype=np.int32)
face_recognizer.train(faces, labels)

# Save the trained model
face_recognizer.save("face_trained_model.xml")