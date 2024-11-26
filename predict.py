import cv2
from ultralytics import YOLO
import torch


# Load the YOLOv8 model
model = YOLO('bestM.pt')  # Adjust the path as necessary
# model = model.to('cuda')
print("deviceeeeeeee: ", model.device)

# print(torch.cuda.is_available())
# print(torch.cuda.current_device())


rtsp_url = "rtsp://admin:webcam123@192.168.1.64/Streaming/channels/101"

# Open the video using OpenCV
video_capture = cv2.VideoCapture(0)

# Iterate over each frame
while video_capture.isOpened():
    ret, frame = video_capture.read()  # Read a frame
    if not ret:
        break
    
    # Apply YOLOv8 object detection
    results = model.track(source=frame,device='cpu')[0]
    
    # Iterate through the detections and draw bounding boxes
    for result in results.boxes.data.tolist():  # Each detection in the format [x1, y1, x2, y2, conf, class]
        x1, y1, x2, y2, conf, cls = result[:6]
        label = f'{model.names[int(cls)]} {conf:.2f}'  # Get class label and confidence
        print("className: ",label, "no: ",int(cls))
        
        # Draw bounding box and label on the frame if confidence is above threshold
        if conf > 0.2: 
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Bounding box
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Label
            
    
    cv2.imshow('webCam',frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
