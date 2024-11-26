import cv2
from ultralytics import YOLO
from vidgear.gears import CamGear
import threading
from face_deep import face_Recoginition

# Load the YOLO models
model_person = YOLO('yolo11n.pt')  # Person detection model
model_ID = YOLO('Version 4 Best.pt')  # Face recognition model

# RTSP stream URL
rtsp_url = "rtsp://admin:webcam123@192.168.1.64/Streaming/channels/101"

# Define video stream options
options = {
    "CAP_PROP_FRAME_WIDTH": 860,
    "CAP_PROP_FRAME_HEIGHT": 480,
    "CAP_PROP_FPS": 30,
}

# Initialize the video stream
stream = CamGear(source='cutv1.mp4', **options).start()

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

# Iterate over each frame
while True:
    with lock:
        frame = frame_buffer
    
    if frame is None:
        continue  # Skip if frame is not available
    
    frame = cv2.resize(frame,(860,480))

    # Apply YOLOv8 person detection
    results = model_person.track(frame, classes=[0])[0]

    # Process detections
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result[:6]
        
        if conf > 0.5:  # Threshold for person detection
            person_frame = frame[int(y1):int(y2), int(x1):int(x2)]
            results_ID = model_ID(person_frame)[0]
            
            if results_ID.boxes.data.tolist():
                for resultID in results_ID.boxes.data.tolist():
                    xt1, yt1, xt2, yt2, conft, clst = resultID[:6]
                    
                    # Skip unwanted classes (0, 1, 2)
                    if clst in [0, 1, 2]:
                        continue
                        
                    if conft > 0.3:  # Threshold for face recognition
                        labelt = f'{model_ID.names[int(clst)]}'
                        cv2.putText(person_frame, labelt, (int(xt1), int(yt1) - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        cv2.rectangle(person_frame, (int(xt1), int(yt1)), (int(xt2), int(yt2)), (0, 255, 0), 2)

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

        # Draw YOLO detection box and label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        label = f'{model_person.names[int(cls)]} {conf:.2f}'
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('webCam', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
stream.stop()
cv2.destroyAllWindows()
