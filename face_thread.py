import cv2
import threading
from deepface import DeepFace
import queue
 
# Initialize video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 
# Path to the database of known faces
db_path = "D:\\Product Development\\Praticing_Folder\\Face_Deep\\images"
 
# Queue to store frames for processing
frame_queue = queue.Queue()
 
# Lock to synchronize access to shared variables
face_lock = threading.Lock()
face_result = None
face_match = False
 
# Function to check for face recognition in the frame
def check_face(frame):
    global face_match, face_result
    try:
        # Perform face recognition using DeepFace
        results = DeepFace.find(frame, db_path=db_path, model_name="Facenet512", enforce_detection=False)
 
        # Check if any results were found
        if results and isinstance(results, list):
            df = results[0]  # The first DataFrame in the list
            if not df.empty and 'identity' in df.columns:  # Ensure the 'identity' column exists and is not empty
                identity = df['identity']
                if not identity.empty:
                    # Extract name from the path in the 'identity' column
                    name = identity.iloc[0].split('\\')[5].split('.')[0]
                    
                    # Get bounding box information
                    x_min = df['source_x'].iloc[0]
                    y_min = df['source_y'].iloc[0]
                    width = df['source_w'].iloc[0]
                    height = df['source_h'].iloc[0]
                    x_max = x_min + width
                    y_max = y_min + height
 
                    # Acquire the lock to safely update shared variables
                    with face_lock:
                        face_match = True
                        face_result = (name, x_min, y_min, x_max, y_max)
    except Exception as e:
        print(f"Error in face recognition: {e}")
 
# Function to process frames from the queue
def process_frames():
    global face_match, face_result
    while True:
        frame = frame_queue.get()  # Block until a frame is available
        if frame is None:
            break  # Exit the loop when None is encountered (sentinel value)
        
        check_face(frame)
 
# Start a thread to process frames
thread = threading.Thread(target=process_frames)
thread.start()
 
# Start capturing video frames
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if count % 30 == 0:  # Process every third frame to reduce computation
            # Put the frame into the queue for processing by the thread
            frame_queue.put(frame.copy())
 
        # Check if a face was detected and recognized
        with face_lock:
            if face_match:
                name, x1, y1, x2, y2 = face_result
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw a rectangle around the face
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)  # Display name
 
        # Show the live video frame
        cv2.imshow('Live', frame)
 
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    count += 1
 
# Stop the processing thread by sending a None value to the queue
frame_queue.put(None)
thread.join()
 
# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()