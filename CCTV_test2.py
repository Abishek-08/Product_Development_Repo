from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from queue import Queue

model = YOLO("yolo11n.pt")
video_path = "cutv1.mp4"
cap = cv2.VideoCapture(video_path)
track_history = defaultdict(lambda: [])
queue =Queue(maxsize=20)

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        resized_frame = cv2.resize(frame,(860,480))
        results = model.track(resized_frame,persist=True,tracker="botsort.yaml",classes=[0],iou=0.7)
        boxes = results[0].boxes.xywh.cpu()
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            resized_frame = results[0].plot()
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box.data.tolist()
                track = track_history[track_id]
                
                if len(track) > 30:
                    track.pop(0)
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(resized_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                    
        cv2.imshow("YOLO11 Tracking", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
             
        
    else:
        break
cap.release()
cv2.destroyAllWindows()