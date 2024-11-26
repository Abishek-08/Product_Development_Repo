import cv2
import threading
from ultralytics import YOLO

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
    video_capture = cv2.VideoCapture(0)  # Use RTSP URL
    while running:
        ret, frame = video_capture.read()
        if not ret:
            break
        with frame_lock:
            # Resize frame to reduce processing time
            frame_queue.append(cv2.resize(frame,(640,480)))  # You can resize here if needed
    video_capture.release()

# Function to process frames
def process_frames():
    global running
    while running:
        if frame_queue:
            with frame_lock:
                frame = frame_queue.pop(0)
            
            # Apply YOLOv8 object detection
            results = model(frame)[0]
            
            # Draw bounding boxes on the frame
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = result[:6]
                label = f'{model.names[int(cls)]} {conf:.2f}'
                print("lable:  ",label,int(cls))
                
                # Draw bounding box and label if confidence is above threshold
                if conf > 0.5 and int(cls)==0:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Store the processed frame back into the queue for displaying
            with frame_lock:
                frame_queue.append(frame)

# Function to display frames
def display_frames():
    while running:
        if frame_queue:
            with frame_lock:
                frame = frame_queue.pop(0)
            cv2.imshow('WebCam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start the threads
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)


capture_thread.start()
process_thread.start()

# Start displaying in the main thread
try:
    display_frames()
finally:
    running = False  # Ensure the threads stop
    capture_thread.join()
    process_thread.join()
    cv2.destroyAllWindows()
