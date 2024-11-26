import cv2
import numpy as np
from pymongo import MongoClient
from ultralytics import YOLO
from vidgear.gears import CamGear
import threading
from face_deep import face_Recoginition

# MongoDB setup
mongo_client = MongoClient('mongodb://localhost:27017/')  # Adjust the URI as needed
db = mongo_client['face2_db']  # Replace with your database name
collection = db['demo']  # Replace with your collection name

# Load the YOLO models
model_person = YOLO('bestP.pt')  # Person detection model
model_ID = YOLO('bestV2.pt')  # Face recognition model

# RTSP stream URL
rtsp_url = "rtsp://admin:webcam123@192.168.1.64/Streaming/channels/101"

# Define video stream options
options = {
    "CAP_PROP_FRAME_WIDTH": 640,
    "CAP_PROP_FRAME_HEIGHT": 480,
    "CAP_PROP_FPS": 30,
}

# Initialize the video stream
stream = CamGear(source=0, **options).start()

# Frame buffer and threading
frame_buffer = None
lock = threading.Lock()

def capture_frames():
    global frame_buffer
    while True:
        frame = stream.read()
        with lock:
            frame_buffer = frame

# Start the frame capture thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

# Function to save frame to MongoDB
def save_frame_to_mongo(frame, label):
    _, buffer = cv2.imencode('.jpg', frame)  # Encode frame as JPEG
    frame_bytes = buffer.tobytes()  # Convert to bytes

    document = {
        'frame': frame_bytes,
        'label': label,
    }

    collection.insert_one(document)  # Insert the document into the collection

# Track detected persons
detected_boxes = set()

# Iterate over each frame
while True:
    with lock:
        frame = frame_buffer
    
    if frame is None:
        continue  # Skip if frame is not available

    # Apply YOLOv8 person detection
    results = model_person.track(frame, classes=[0])[0]

    # Process detections
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result[:6]
        
        if conf > 0.8:  # Threshold for person detection
            person_frame = frame[int(y1):int(y2), int(x1):int(x2)]
            person_box = (int(x1), int(y1), int(x2), int(y2))

            # Check if this box is already detected
            if person_box not in detected_boxes:
                detected_boxes.add(person_box)  # Add the box to the set

                results_ID = model_ID.track(person_frame)[0]
                
                if results_ID.boxes.data.tolist():
                    for resultID in results_ID.boxes.data.tolist():
                        xt1, yt1, xt2, yt2, conft, clst = resultID[:6]
                        
                        print(f'Detected ID Class: {int(clst)} with confidence: {conft}')

                        # Check for different ID cards and save to MongoDB
                        if int(clst) == 0:
                            print("Person with Blue IdCard")
                            cv2.putText(frame, "with ID", (int(x1), int(y1) - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            # save_frame_to_mongo(person_frame, "Blue IdCard")
                            
                        elif int(clst) == 1:
                            print("Person with Green IdCard")
                            cv2.putText(frame, "with ID", (int(x1), int(y1) - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            # save_frame_to_mongo(person_frame, "Green IdCard")
                            
                        elif int(clst) == 2:
                            print("Person with Red IdCard")
                            cv2.putText(frame, "with ID", (int(x1), int(y1) - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            # save_frame_to_mongo(person_frame, "Red IdCard")

                else:
                    print("Person without IdCard")
                    cv2.putText(frame, "without ID", (int(x1), int(y1) - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    
                    # Perform face recognition
                    face_roi = frame[int(y1):int(y2), int(x1):int(x2)]
                    face_recognition_result = face_Recoginition(face_roi)

                    if face_recognition_result is not None:
                        name, x_min, y_min, x_max, y_max = face_recognition_result
                        # Convert face recognition coordinates to original frame coordinates
                        x_min_original = int(x1) + int(x_min)
                        y_min_original = int(y1) + int(y_min)
                        x_max_original = int(x1) + int(x_max)
                        y_max_original = int(y1) + int(y_max)

                        # Draw face recognition box and name in original frame coordinates
                        cv2.rectangle(frame, (x_min_original, y_min_original), (x_max_original, y_max_original), (0, 255, 255), 2)
                        cv2.putText(frame, name, (x_min_original, y_min_original - 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                        
                        save_frame_to_mongo(face_roi, "Without IdCard")
                        

    cv2.imshow('webCam', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
stream.stop()
cv2.destroyAllWindows()
