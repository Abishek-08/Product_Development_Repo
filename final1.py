import cv2
from ultralytics import YOLO
from collections import defaultdict
from queue import Queue
import threading
 
# Load YOLO model
model = YOLO('bestFinal.pt')
 
# Open video capture
cap = cv2.VideoCapture('cutv1.mp4')
 
# Count for frame skipping
count = 0
 
# Store the track history
track_history = defaultdict(lambda: [])
 
# Function to check if the bounding boxes of the person and the blue tag overlap
def is_overlapping(box1, box2):
    # Get coordinates for box1 (person) and box2 (blue tag)
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to absolute coordinates
    x1_min = x1 - w1 / 2
    y1_min = y1 - h1 / 2
    x1_max = x1 + w1 / 2
    y1_max = y1 + h1 / 2
    
    x2_min = x2 - w2 / 2
    y2_min = y2 - h2 / 2
    x2_max = x2 + w2 / 2
    y2_max = y2 + h2 / 2
    
    # Check if the bounding boxes overlap
    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)
 
# Main loop to process the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    count += 1  # Skip frames to reduce processing load
    if count % 5 != 0:  # Adjust the frame skip value as needed
        continue
 
    frame_resized = cv2.resize(frame, (860, 480))
 
    # Run YOLOv5 tracking on the frame, persisting tracks between frames
    results = model.track(frame_resized, persist=True)
 
    class_ids = results[0].boxes.cls.int().cpu().tolist()  # Get the class ids
    boxes = results[0].boxes.xywh.cpu()  # Get the bounding boxes
    track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
 
    # Store the person boxes and blue tag boxes
    person_boxes = []
    blue_tag_boxes = []
 
    # Process each detected object
    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
        # Case 1: Person detection (class_id == 2)
        if int(class_id) == 2:
            x, y, w, h = box
            person_boxes.append((x, y, w, h, track_id))  # Store the person box and track_id
 
        # Case 2: Blue tag detection (class_id == 0)
        elif int(class_id) == 0:
            x, y, w, h = box
            blue_tag_boxes.append((x, y, w, h, track_id))  # Store the blue tag box and track_id
 
    # Now, check if any person is wearing a blue tag (overlapping bounding boxes)
    for person in person_boxes:
        x, y, w, h, person_track_id = person
        for blue_tag in blue_tag_boxes:
            x_tag, y_tag, w_tag, h_tag, blue_tag_track_id = blue_tag
 
            # Check if the person's bounding box overlaps with the blue tag's bounding box
            if is_overlapping((x, y, w, h), (x_tag, y_tag, w_tag, h_tag)):
                # If they overlap, consider the person as wearing a blue tag
                cv2.rectangle(frame_resized, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
                cv2.putText(frame_resized, f"Person with Blue Tag Track: {person_track_id}", (int(x - w / 2), int(y - h / 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame_resized, f"Class: {class_id}", (int(x - w / 2), int(y - h / 2) - 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
 
    # Display the frame with annotations
    cv2.imshow("Video", frame_resized)
 
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Release video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()