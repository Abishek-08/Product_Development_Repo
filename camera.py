import cv2
from deepface import DeepFace
from insertDB import findEmployee
import threading

# Global variables
frame_to_process = None
running = True
frame = None  # For access in the analyze function

# Function to analyze frames in a separate thread
def analyze_frame():
    global frame_to_process, frame
    while running:
        if frame_to_process is not None:
            # Analyze the frame for faces
            results = DeepFace.find(frame_to_process, db_path="D:\Product Development\Praticing_Folder\Face_Deep\images", model_name="Facenet512", enforce_detection=False)
            process_results(results)
            frame_to_process = None  # Clear the frame after processing

def process_results(results):
    global frame  # Access the current frame for drawing bounding boxes
    if results:
        for result in results:
            identity = result.get('identity', None)
            if identity is not None and not identity.empty:
                name = identity[0].split('\\')[5].split('.')[0]  # Adjust path handling

                # Get bounding box coordinates
                x_min = result.get('source_x', [0])[0]
                y_min = result.get('source_y', [0])[0]
                height = result.get('source_h', [0])[0]
                width = result.get('source_w', [0])[0]


                x_max = x_min + width
                y_max = y_min + height

                # Debug output
                print(f"Detected {name} at: ({x_min}, {y_min}, {x_max}, {y_max})")

                # Draw rectangle around the face and label it
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                detectedEmployee = findEmployee(name)
                print("The Detected Employee Details is: ", detectedEmployee)

# Start the analysis thread
analysis_thread = threading.Thread(target=analyze_frame)
analysis_thread.start()

rtsp_url = "rtsp://admin:webcam123@192.168.1.64/Streaming/channels/101"

webcam = cv2.VideoCapture(rtsp_url)

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Failed to capture video")
        break

    # Resize the frame for faster processing
    # frame_small = cv2.resize(frame, (320, 240))  # Reduce resolution

    # Every 5th frame, set it for analysis
    if int(webcam.get(cv2.CAP_PROP_POS_FRAMES)) % 5 == 0:
        frame_to_process = frame

    # Show the video feed with detected faces
    cv2.imshow("live", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

# Clean up
webcam.release()
cv2.destroyAllWindows()
