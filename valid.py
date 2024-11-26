# import cv2
# from deepface import DeepFace

# webcam = cv2.VideoCapture(0)

# while True:
#     ret,frame = webcam.read()
    
#     result = DeepFace.find(frame,db_path = "D:\Product Development\Face_Deep\images", model_name = "Facenet512", enforce_detection=False)
    
#     if len(result[0]['identity']) > 1:
#         name = result[0]['identity'][1].split('\\')[4].split('.')[0]
#         x_min = result[0]['source_x'][0]
#         y_min = result[0]['source_y'][0]
        
#         height = result[0]['source_h'][0]
#         width = result[0]['source_w'][0]
        
#         x_max = (x_min+width)
#         y_max = (y_min+height)
        
#         cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(0,0,255),2)
#         cv2.putText(frame,name,(x_min,y_min),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4)
    
    
#     cv2.imshow("live",frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
# webcam.release()
# cv2.destroyAllWindows()



import cv2
from deepface import DeepFace

# Initialize webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Find faces in the frame
    results = DeepFace.find(frame, db_path="D:\Product Development\Face_Deep\images", model_name="Facenet512", enforce_detection=False)

    print("firstzzzzz: ",results[0])

    # Check if any results were found
    if len(results[0]) > 1:
        for result in results:
            print("resultsss: ",result['identity'])
            print("length: ",len(result['identity']))
            if len(result['identity']) > 1:
                # Extract face coordinates and name
                name = result['identity'][0].split('\\')[4].split('.')[0]  # Adjusted for file path
                x_min = result['source_x'][0]
                y_min = result['source_y'][0]
                width = result['source_w'][0]
                height = result['source_h'][0]

                x_max = x_min + width
                y_max = y_min + height
            
                # Draw rectangle and put text on the frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(frame, name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
    else:
        print("No faces detected or no matching identities found.")

    # Display the frame with detected faces
    cv2.imshow("Live", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
webcam.release()
cv2.destroyAllWindows()


