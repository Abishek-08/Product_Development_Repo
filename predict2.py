import cv2
import threading
from ultralytics import YOLO
import time

# Load the YOLOv8 model
model = YOLO('yolo11n.pt')  # Adjust the path as necessary

rtsp_url = "rtsp://admin:webcam123@192.168.1.64/Streaming/channels/101"

# Initialize a thread-safe queue for frames
frame_queue = []
frame_lock = threading.Lock()
running = True

# Function to capture video frames
def capture_frames():
    global running
    video_capture = cv2.VideoCapture(rtsp_url)  # Use RTSP URL
    video_capture.set(cv2.CAP_PROP_FPS, 15)  # Set lower frame rate if needed
    while running:
        ret, frame = video_capture.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (320, 240))  # Resize to a smaller resolution
        with frame_lock:
            if len(frame_queue) < 10:  # Limit the size of the queue
                frame_queue.append(resized_frame)
    video_capture.release()

# Function to process frames
def process_frames():
    global running
    while running:
        with frame_lock:
            if frame_queue:
                frame = frame_queue.pop(0)
            else:
                frame = None

        if frame is not None:
            # Apply YOLOv8 object detection
            results = model(frame)[0]

            # Draw bounding boxes on the frame
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = result[:6]
                label = f'{model.names[int(cls)]} {conf:.2f}'
                
                # Draw bounding box and label if confidence is above threshold
                if conf > 0.5:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display the processed frame
            cv2.imshow('WebCam', frame)
        
        # Exit condition for quitting
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

# Start the threads
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)

capture_thread.start()
process_thread.start()

# Wait for threads to finish
capture_thread.join()
process_thread.join()
cv2.destroyAllWindows()
