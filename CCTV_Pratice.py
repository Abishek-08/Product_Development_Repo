from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolov8n.pt")
model_ID = YOLO('bestV2.pt')

# Open the video file
video_path = "cutv1.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])



# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        resized_frame = cv2.resize(frame,(860,480))
        results = model.track(resized_frame, persist=True,tracker="botsort.yaml",classes=[0])

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        if results[0].boxes.is_track:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        
            # Visualize the results on the frame
            annotated_frame = results[0].plot()

        
            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                
                # Calculate the top-left and bottom-right corner points
                x_min = x - w / 2
                y_min = y - h / 2
                x_max = x + w / 2
                y_max = y + h / 2
 
                # Crop the frame using the bounding box coordinates
                cropped_frame = annotated_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                
                results_ID = model_ID.predict(cropped_frame)
                for result_ID in results_ID[0].boxes.data.tolist():
                    xt1,yt1,xt2,yt2,conft,clst = result_ID[:6]
                    
                    cv2.rectangle(annotated_frame,((int(xt1)+int(x_min)),(int(yt1)+int(y_min))),((int(xt2)+int(x_max)),(int(y_max)+int(yt2))),(0,0,255),2)
                    labelt = f'{model_ID.names[int(clst)]}'
                    cv2.putText(annotated_frame,labelt,((int(xt1)+int(x_min)),(int(yt1)+int(y_min))),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
 
                    
                   
        
            # Display the annotated frame
            cv2.imshow("YOLO11 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
               break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()