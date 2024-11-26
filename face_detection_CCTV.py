import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
from tracker import *
from face_thread import check_face
import threading
from queue import Queue

model = YOLO('yolo11n.pt')
video = cv2.VideoCapture('chennai.mp4')

count = 0
tracker = Tracker()
cropped_person_queue = Queue(maxsize=20)

def checking_face():
    while True:
        item = cropped_person_queue.get()
        if item is not None:
            face_recognition_result = check_face(item)
            if face_recognition_result is not None:
                name, x_min, y_min, x_max, y_max = face_recognition_result
                frame_coords.put((name, x_min, y_min, x_max, y_max))
        cropped_person_queue.task_done()

frame_coords = Queue()

threading.Thread(target=checking_face, daemon=True).start()

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    count += 1
    if count % 30 == 0:
        continue

    frame = cv2.resize(frame, (860, 480))
    results = model.predict(frame, classes=[0])[0]
    a = results.boxes.data
    px = pd.DataFrame(a).astype('float')
    list_bboxes = []

    for index, row in px.iterrows():
        x1, y1, x2, y2 = map(int, row[:4])
        list_bboxes.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list_bboxes)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
        cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cropped_person = frame[y1:y2, x1:x2]
        height, width = cropped_person.shape[:2]
        cropped = cropped_person[:height//2, :]
        kernel = np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]])
        bright = cv2.addWeighted(cropped, 1.8, cropped, 0, 38)

        if not cropped_person_queue.full():
            cropped_person_queue.put(bright)

    while not frame_coords.empty():
        name, x_min, y_min, x_max, y_max = frame_coords.get()
        x_min_original = int(x3) + int(x_min)
        y_min_original = int(y3) + int(y_min)
        x_max_original = int(x3) + int(x_max)
        y_max_original = int(y3) + int(y_max)
        cv2.rectangle(frame, (x_min_original, y_min_original), (x_max_original, y_max_original), (0, 255, 255), 2)
        cv2.putText(frame, name, (x_min_original, y_min_original - 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
