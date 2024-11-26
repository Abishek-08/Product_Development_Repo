import cv2
from ultralytics import YOLO
from vidgear.gears import VideoGear
import threading
from face_deep import face_Recoginition
from ValidFaceDB import findEmployee

# Load the YOLOv8 model
model = YOLO('yolo11n.pt')  # Adjust the path as necessary
# model = model.to('cuda')

rtsp_url = "rtsp://admin:webcam123@192.168.1.64/Streaming/channels/101"

# Define suitable tweak parameters for your stream.
options = {
    "CAP_PROP_FRAME_WIDTH": 640,  # Reduce resolution
    "CAP_PROP_FRAME_HEIGHT": 480,
    "CAP_PROP_FPS": 60,
}

# Initialize the video stream
stream = VideoGear(source=0, **options).start()

# To manage threading
frame_buffer = None
lock = threading.Lock()

def capture_frames():
    global frame_buffer
    while True:
        frame = stream.read()
        with lock:
            frame_buffer = frame

# Start a thread for capturing frames
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

# Iterate over each frame
while True:
    with lock:
        frame = frame_buffer
    
    if frame is None:
        continue  # Skip if frame is not available

    # Apply YOLOv8 object detection
    results = model(frame)[0]

    # Iterate through the detections
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result[:6]
        
        if conf > 0.5 and int(cls) == 0:  # Only consider the class you want
            # Crop the detected face area for recognition
            face_roi = frame[int(y1):int(y2), int(x1):int(x2)]
            face_recognition_result = face_Recoginition(face_roi)

            if face_recognition_result is not None:
                name, x_min, y_min, x_max, y_max = face_recognition_result
                
                print("Detected Employee Details from DB: ",findEmployee(name))
                
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
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
           
            
    
    cv2.imshow('webCam', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
stream.stop()
cv2.destroyAllWindows()
