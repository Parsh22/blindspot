import cv2
import face_recognition
import pickle
from speech2text import *
from circuit import *


def authenticate(frame):




    try:
        with open('face_data.dat', 'rb') as f:
            known_face_data = pickle.load(f)
        known_face_encodings = known_face_data['encodings']
        known_face_labels = known_face_data['labels']
    except FileNotFoundError:
        known_face_encodings = []
        known_face_labels = []

    # Find all the faces and their encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the current face encoding with the known face encodings
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding)

        # If a match is found, display a welcome message and return True
        if True in matches:
            label = known_face_labels[matches.index(True)]
            return label

    return None


def sign_up(frame):

    try:
        with open('face_data.dat', 'rb') as f:
            known_face_data = pickle.load(f)
        known_face_encodings = known_face_data['encodings']
        known_face_labels = known_face_data['labels']
    except FileNotFoundError:
        known_face_encodings = []
        known_face_labels = []

    # Find all the faces and their encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Prompt user to enter a label for the face
    if len(face_encodings) > 0:
        cv2.imshow('Video', frame)
        label = input("Enter label: ")
        known_face_encodings.append(face_encodings[0])
        known_face_labels.append(label)

        # Save known face encodings and corresponding labels to file
        with open('face_data.dat', 'wb') as f:
            pickle.dump({'encodings': known_face_encodings,
                        'labels': known_face_labels}, f)

    # No need for cap.release() and cv2.destroyAllWindows() as they are handled in yolo.py


sign_up()
