import cv2
from ultralytics import YOLO
from bleak import BleakClient
import numpy as np
import pygame


# pygame.init()
# pygame.mixer.music.load("D:\Product Development\Object_Bottle_Buds\Calling-Santa.mp3")
# pygame.mixer.music.play()
# address = "8C:64:A2:8A:A7:4E"

# Load the YOLOv8 model
model = YOLO('./runs/detect/train3/weights/best.pt')  # Adjust the path as necessary

# Open the video using OpenCV
video_capture = cv2.VideoCapture(0)

# Iterate over each frame
while video_capture.isOpened():
    ret, frame = video_capture.read()  # Read a frame
    if not ret:
        break
    
    # Clone the original frame for displaying without detection
    original_frame = frame.copy()
    
    # Apply YOLOv8 object detection
    results = model(frame)[0]
    
    # Iterate through the detections and draw bounding boxes
    for result in results.boxes.data.tolist():  # Each detection in the format [x1, y1, x2, y2, conf, class]
        x1, y1, x2, y2, conf, cls = result[:6]
        label = f'{model.names[int(cls)]} {conf:.2f}'  # Get class label and confidence
        
        # Draw bounding box and label on the frame if confidence is above threshold
        if conf > 0.5: 
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Bounding box
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Label
            
    
    # # Resize frames for better viewing
    # original_frame = cv2.resize(original_frame, (320, 240))
    # frame = cv2.resize(frame, (320, 240))
    
    # # Stack both frames side by side
    # combined_frame = np.hstack((original_frame, frame))
    
    # Display the combined frame
    #cv2.imshow('Live Webcam Feed - Left: Original, Right: Detection', combined_frame)
    cv2.imshow('webCam',frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
