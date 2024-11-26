import threading
import cv2
from ultralytics import YOLO
import os
from queue import Queue

# Define model names and video sources
MODEL_NAMES = ["yolo11n.pt", "yolo11n-seg.pt"]
SOURCES = ["cutv1.mp4"]  # local video, 0 for webcam
person_queue = Queue(maxsize=20)

def run_tracker_in_thread(model_name, filename):
    
    model = YOLO(model_name)
    model_ID = YOLO('bestV2.pt')
    
    video = cv2.VideoCapture(os.path.join(filename))
    
    ret = True
    
    while ret:
        ret,frame = video.read()
    
        resized_Frame =  cv2.resize(frame,(860,480))

        results = model_ID.track(source=resized_Frame,tracker="botsort.yaml")[0]

        if results:
            for result in results.boxes.data.tolist():
                x1,y1,x2,y2,cnf,cls = result[:6]
                
                cropped_frame = resized_Frame[int(y1):int(y2),int(x1):int(x2)]
                person_queue.put(cropped_frame)
                
                if not person_queue.full():
                    # cv2.imshow('crop',person_queue.get())
                    
                    results_ID = model_ID.predict(person_queue.get())[0]
                    
                    if results_ID.boxes.data.tolist():
                        for result in results_ID.boxes.data.tolist():
                            xt1,yt1,xt2,yt2,conft,clst = result[:6]
                            
                            cv2.rectangle(resized_Frame,((int(x1)-int(xt1)),(int(y1)-int(yt1))),((int(x2)-int(xt2)),(int(y2)-int(yt2))),(255,0,0),2)
                            lablet = f'{model_ID.names[int(clst)]}'
                            cv2.putText(resized_Frame,lablet,((int(x1)-int(xt1)),(int(y1)-int(yt1))),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
                        
        
                cv2.rectangle(resized_Frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
                label = f'{model_ID.names[int(cls)]}'
                cv2.putText(resized_Frame,label,(int(x1),int(y1)-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow("video", resized_Frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

    
    

# Create and start tracker threads using a for loop
tracker_threads = []
for video_file, model_name in zip(SOURCES, MODEL_NAMES):
    thread = threading.Thread(target=run_tracker_in_thread, args=(model_name, video_file), daemon=True)
    tracker_threads.append(thread)
    thread.start()
    

# Wait for all tracker threads to finish
for thread in tracker_threads:
    thread.join()
    

# Clean up and close windows
cv2.destroyAllWindows()