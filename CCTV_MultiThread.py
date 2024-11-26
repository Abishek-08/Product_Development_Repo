from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Open the video file
video_path = "cutv1.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

region_area = [(580,219),(700,261),(707,240),(597,201)]

count = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        resized_frame = cv2.resize(frame,(860,480))
        results = model.track(resized_frame, persist=True,classes=[0])

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        if results[0].boxes.is_track:
           track_ids = results[0].boxes.id.int().cpu().tolist()

           # Visualize the results on the frame
           annotated_frame = results[0].plot()
           
           print("Track IDDDD: ",track_ids)
           

           # Plot the tracks
           for box, track_id in zip(boxes, track_ids):
               x, y, w, h = box
               track = track_history[track_id]
               track.append((float(x), float(y)))  # x, y center point
               
               print("Zipppppp: ","  boxess: ",box,"  trackId: ",track_id)
               
               # Calculate the top-left and bottom-right corner points
               x_min = x - w / 2
               y_min = y - h / 2
               x_max = x + w / 2
               y_max = y + h / 2
               
               cv2.circle(annotated_frame,(int(x),int(y)),1,(0,255,0),2)
               
               cv2.polylines(annotated_frame,[np.array(region_area,np.int32)],True,(255,0,0),2)
               
               result = cv2.pointPolygonTest(np.array(region_area,np.int32),(int(x),int(y)),False)
               
               if result > 0:
                   count = count+1
                   print("person crossing")
               
               if len(track) > 30:  # retain 90 tracks for 90 frames
                  track.pop(0)
                  
                  
                  
                  
                  

           # Display the annotated frame
           disp_count = f'{"count ",count}'
           cv2.putText(annotated_frame,disp_count,(70,70),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
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