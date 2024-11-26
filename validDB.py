import cv2
from deepface import DeepFace
from insertDB import findEmployee

webcam = cv2.VideoCapture(0)

while True:
    ret,frame = webcam.read()
    
    result = DeepFace.find(frame,db_path = "D:\Product Development\Face_Deep\images", model_name = "Facenet512", enforce_detection=False)
    
    if len(result[0]['identity']) > 1:
        name = result[0]['identity'][1].split('\\')[4].split('.')[0]
        x_min = result[0]['source_x'][0]
        y_min = result[0]['source_y'][0]
        
        height = result[0]['source_h'][0]
        width = result[0]['source_w'][0]
        
        x_max = (x_min+width)
        y_max = (y_min+height)
        
        cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(0,0,255),2)
        cv2.putText(frame,name,(x_min,y_min),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4)
        
        detectedEmployee = findEmployee(name)
        print("The Detected Employee Details is: ",detectedEmployee)
    
    cv2.imshow("live",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
webcam.release()
cv2.destroyAllWindows()