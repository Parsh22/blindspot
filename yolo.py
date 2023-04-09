import pickle
import face_recognition
import cv2
import serial
import numpy as np
from speech2text import *
from test_automation import *

# Open live stream
cap = cv2.VideoCapture(1)


def sign_in():
    try:
        with open('face_data.dat', 'rb') as f:
            known_face_data = pickle.load(f)
        known_face_encodings = known_face_data['encodings']
        known_face_labels = known_face_data['labels']
    except FileNotFoundError:
        known_face_encodings = []
        known_face_labels = []

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Find all the faces and their encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the current face encoding with the known face encodings
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding)

        # If a match is found, display a welcome message and stop capturing video
        if True in matches:
            label = known_face_labels[matches.index(True)]
            print(label + " detected")
            break

    # Display the resulting image
    cv2.imshow('Video', frame)
    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return


def distance_calculate():
    # Create a serial connection to the Arduino
    ser = serial.Serial('COM5', 9600)

    # Loop indefinitely, reading data from the Arduino
    # Read a line of data from the Arduino
    data = ser.readline().strip()

    # Decode the data as a string
    data_str = data.decode('utf-8')
    return data_str


# Load YOLOv4 object detection model
net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Define classes
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
last_label = ''

# Initialize variables for tracking objects
tracked_objects = {}
object_id = 0

while True:
    # Read frame from live stream
    ret, frame = cap.read()

    # Perform object detection
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Parse output and draw bounding boxes
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.9 and classes[class_id] in classes:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and class labels
    font = cv2.FONT_HERSHEY_PLAIN
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        label = classes[class_ids[i]]
        confidence = confidences[i]
        if confidence > 0.9 and label != last_label:
            last_label = label
            tracked_objects[label] = object_id
            object_id += 1
            # distance = np.random.randint(1, 10) # Replace with actual distance calculation
            if label == 'person':
                sign_in()
            distance = distance_calculate()  # Replace with actual distance calculation
            # distance = 78
            input_str = f"A {label} was detected {distance} centimeters away."
            print(input_str)
            speak(input_str, 150)
        color = (0, 255, 0)
        cv2.rectangle(frame, (left, top),
                      (left + width, top + height), color, 2)
        cv2.putText(frame, label, (left, top - 5), font, 1, color, 1)

    # Show frame
    cv2.imshow('Object detection', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
browser_call()
