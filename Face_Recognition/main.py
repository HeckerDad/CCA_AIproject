import cv2
import numpy as np

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('C:\\Users\\Hp\\Desktop\\Face_Recognition\\haarcascade_frontalface_default.xml')

# Load the pre-trained face recognition model
model = cv2.face.LBPHFaceRecognizer_create()
model.read('C:\\Users\\Hp\\Desktop\\Face_Recognition\\face_trained_model.xml')

# Define a function to recognize faces in an image
def recognize_faces(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop over each face in the image
    for (x, y, w, h) in faces:
        # Extract the face from the image
        face = gray[y:y+h, x:x+w]
        
        # Resize the face to the required size
        face = cv2.resize(face, (96, 96))
        
        # Normalize the pixel values of the face
        face = face / 255.0
        
        # Use the model to predict the identity of the face
        identity, confidence = model.predict(face)
        
        # Print the predicted identity and confidence
        print('Identity:', identity, 'Confidence:', confidence)
        
        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if identity <= 5:
            identity = str(identity)
            identity = "Naitik"
            with open('C:\\Users\\Hp\\Desktop\\AI\\commondata.txt', 'w') as file:
                file.truncate(0)
                file.write("Naitik")
        
        #TODO: Implement this
        # if (confidence <= 60) or (len(faces) == 0):
        #     identity = str(identity)
        #     identity = "Unknown"
        #     with open('C:\\Users\\Hp\\Desktop\\AI\\commondata.txt', 'w') as file:
        #         file.truncate(0)
        #         file.write("Unkown")

        # Add a label to the rectangle
        cv2.putText(image, str(identity), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image

# Open the video stream
cap = cv2.VideoCapture(0)

# Loop over each frame in the video stream
while True:
    # Read the next frame from the video stream
    ret, frame = cap.read()
    
    # Recognize faces in the frame
    frame = recognize_faces(frame)
    
    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Wait for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #Empty commandata
        with open("commondata.txt", "w") as f:
            f.truncate(0)
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()