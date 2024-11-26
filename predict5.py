import cv2
from ultralytics import YOLO
from vidgear.gears import CamGear,VideoGear


# Load the YOLOv8 model
model = YOLO('yolo11n.pt')  # Adjust the path as necessary

rtsp_url = "rtsp://admin:webcam123@192.168.1.64/Streaming/channels/102"

# define suitable tweak parameters for your stream.
options = {
    "CAP_PROP_FRAME_WIDTH": 320, # resolution 320x240
    "CAP_PROP_FRAME_HEIGHT": 240,
    "CAP_PROP_FPS": 30, # framerate 60fps
}

# Initialize the video stream
stream = VideoGear(source=rtsp_url,**options).start()

# Open the video using OpenCV
# video_capture = cv2.VideoCapture(rtsp_url)

# Iterate over each frame
while True:
    frame = stream.read()  # Read a frame
 
    
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
            
    
    cv2.imshow('webCam',frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
stream.stop()
cv2.destroyAllWindows()





