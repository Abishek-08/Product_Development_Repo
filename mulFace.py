import cv2
from deepface import DeepFace
# from insertDB import findEmployee

webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()

    # Analyze the frame for multiple faces
    results = DeepFace.find(frame, db_path="D:\Product Development\Implement_Folder\CCTV_Test\images", model_name="Facenet512", enforce_detection=False)

    # Check if any faces were found
    if results:
        for result in results:
            identity = result.get('identity', None)
            if identity is not None and not identity.empty:
                print(identity[0].split('\\')[4])
                name = identity[0].split('\\')[5].split('.')[0]
                x_min = result.get('source_x', [0])[0]
                y_min = result.get('source_y', [0])[0]
                height = result.get('source_h', [0])[0]
                width = result.get('source_w', [0])[0]

                x_max = x_min + width
                y_max = y_min + height

                # Draw rectangle around the face and label it
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(frame, name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # detectedEmployee = findEmployee(name)
                # print("The Detected Employee Details is: ",detectedEmployee)

    # Show the video feed with detected faces
    cv2.imshow("live", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
