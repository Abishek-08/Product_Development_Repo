import cv2
from pymongo import MongoClient
from ultralytics import YOLO
import datetime
import os
import time
from vidgear.gears import VideoGear
from face_deep import face_Recoginition


# enforce UDP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"


class Reconnecting_VideoGear:
    def __init__(self, cam_address, stabilize=False, reset_attempts=50, reset_delay=5, back_end=cv2.CAP_FFMPEG,resolution=(640,480),frame_rate=60):
        self.cam_address = cam_address
        self.stabilize = stabilize
        self.reset_attempts = reset_attempts
        self.reset_delay = reset_delay
        self.back_end = back_end
        self.resolution = resolution
        self.frame_rate = frame_rate
        self.source = VideoGear(
            source=self.cam_address, stabilize=self.stabilize,backend=self.back_end,resolution=self.resolution,framerate=self.frame_rate
        ).start()
        self.running = True

    def read(self):
        if self.source is None:
            return None
        if self.running and self.reset_attempts > 0:
            frame = self.source.read()
            if frame is None:
                self.source.stop()
                self.reset_attempts -= 1
                print(
                    "Re-connection Attempt-{} occured at time:{}".format(
                        str(self.reset_attempts),
                        datetime.datetime.now().strftime("%m-%d-%Y %I:%M:%S%p"),
                    )
                )
                time.sleep(self.reset_delay)
                self.source = VideoGear(
                    source=self.cam_address, stabilize=self.stabilize
                ).start()
                # return previous frame
                return self.frame
            else:
                self.frame = frame
                return frame
        else:
            return None

    def stop(self):
        self.running = False
        self.reset_attempts = 0
        self.frame = None
        if not self.source is None:
            self.source.stop()


if __name__ == "__main__":
    # open any valid video stream
    stream = Reconnecting_VideoGear(
        cam_address='rtsp://admin:webcam123@192.168.1.64/Streaming/channels/101',
        reset_attempts=20,
        reset_delay=5,
        back_end=cv2.CAP_FFMPEG,
        resolution=(640,480),
        frame_rate = 60
    )


# MongoDB setup
mongo_client = MongoClient('mongodb://localhost:27017/')  # Adjust the URI as needed
db = mongo_client['face2_db']  # Replace with your database name
collection = db['demo']  # Replace with your collection name

# Load the YOLO models
model_person = YOLO('bestP.pt')  # Person detection model
model_ID = YOLO('bestV2.pt')  # Face recognition model




# Function to save frame to MongoDB
def save_frame_to_mongo(frame, label):
    _, buffer = cv2.imencode('.jpg', frame)  # Encode frame as JPEG
    frame_bytes = buffer.tobytes()  # Convert to bytes

    document = {
        'frame': frame_bytes,
        'label': label,
    }

    collection.insert_one(document)  # Insert the document into the collection

# Track detected persons
detected_boxes = set()

# Iterate over each frame
while True:
    frame = stream.read()
    
    if frame is None:
        continue  # Skip if frame is not available
    
    resized_frame = cv2.resize(frame,(640,480))

    # Apply YOLOv8 person detection
    results = model_person.track(resized_frame, classes=[0])[0]

    # Process detections
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result[:6]
        
        if conf > 0.5:  # Threshold for person detection
            person_frame = resized_frame[int(y1):int(y2), int(x1):int(x2)]
            person_box = (int(x1), int(y1), int(x2), int(y2))

            # Check if this box is already detected
            if person_box not in detected_boxes:
                detected_boxes.add(person_box)  # Add the box to the set

                results_ID = model_ID.track(person_frame)[0]
                
                if results_ID.boxes.data.tolist():
                    for resultID in results_ID.boxes.data.tolist():
                        xt1, yt1, xt2, yt2, conft, clst = resultID[:6]
                        
                        print(f'Detected ID Class: {int(clst)} with confidence: {conft}')

                        # Check for different ID cards and save to MongoDB
                        if int(clst) == 0 or int(clst)==1 or int(clst)==2:
                            
                            cv2.putText(resized_frame, "with ID", (int(x1), int(y1) - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                            cv2.rectangle(resized_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            # save_frame_to_mongo(person_frame, "Blue IdCard")
                            
                       

                else:
                    print("Person without IdCard")
                    cv2.putText(resized_frame, "without ID", (int(x1), int(y1) - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(resized_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    
                    # Perform face recognition
                    face_roi = resized_frame[int(y1):int(y2), int(x1):int(x2)]
                    face_recognition_result = face_Recoginition(face_roi)

                    if face_recognition_result is not None:
                        name, x_min, y_min, x_max, y_max = face_recognition_result
                        # Convert face recognition coordinates to original frame coordinates
                        x_min_original = int(x1) + int(x_min)
                        y_min_original = int(y1) + int(y_min)
                        x_max_original = int(x1) + int(x_max)
                        y_max_original = int(y1) + int(y_max)

                        # Draw face recognition box and name in original frame coordinates
                        cv2.rectangle(resized_frame,(x_min_original, y_min_original), (x_max_original, y_max_original), (0, 255, 255), 2)
                        cv2.putText(resized_frame, name, (x_min_original, y_min_original - 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
                        
                        # save_frame_to_mongo(face_roi, "Without IdCard")
                        

    cv2.imshow('webCam', resized_frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
stream.stop()
cv2.destroyAllWindows()
