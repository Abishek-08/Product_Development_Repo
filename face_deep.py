import cv2
from deepface import DeepFace

def face_Recoginition(frame):
    # Analyze the frame for multiple faces
    results = DeepFace.find(frame, db_path="D:\Product Development\Implement_Folder\Face_Yolo\images", model_name="Facenet512", enforce_detection=False)
    
    # Check if any faces were found
    if results:
        for result in results:
            identity = result.get('identity', None)
            if identity is not None and not identity.empty:
                print("nameeeeeeee: ", identity[0].split('\\')[5].split('.')[0])
                name = identity[0].split('\\')[5].split('.')[0]
                x_min = result.get('source_x', [0])[0]
                y_min = result.get('source_y', [0])[0]
                height = result.get('source_h', [0])[0]
                width = result.get('source_w', [0])[0]

                x_max = x_min + width
                y_max = y_min + height
                
                if name is not None:
                    # Process the recognized face
                    print(f"Recognized {name} at coordinates ({x_min}, {y_min}, {x_max}, {y_max})")
                    return name,x_min,y_min,x_max,y_max
                else:
                    print("No faces found.")
                    return None,None,None,None,None
    


