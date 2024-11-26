import cv2
from ultralytics import YOLO
from vidgear.gears import VideoGear
import threading


# Load the YOLOv8 model
model = YOLO('bestV2.pt')  # Adjust the path as necessary

rtsp_url = "rtsp://admin:webcam123@192.168.1.64/Streaming/channels/101"

# Define suitable tweak parameters for your stream.
options = {
    "CAP_PROP_FRAME_WIDTH": 640,  # resolution 320x240
    "CAP_PROP_FRAME_HEIGHT": 480,
    "CAP_PROP_FPS": 30,  # framerate 30fps
}

# Initialize the video stream
stream = VideoGear(source=rtsp_url, **options).start()

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
    results = model.track(frame)[0]
    
    # Iterate through the detections and draw bounding boxes
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result[:6]
        label = f'{model.names[int(cls)]} {conf:.2f}'
        
        if conf > 0.5:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('webCam', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
stream.stop()
cv2.destroyAllWindows()