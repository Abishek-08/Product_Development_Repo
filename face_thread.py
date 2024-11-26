import cv2

from deepface import DeepFace

 

 
# Path to the database of known faces
db_path = "D:\Product Development\Implement_Folder\CCTV_Test\images"
 

 
# Function to check for face recognition in the frame
def check_face(frame):
    
    try:
        # Perform face recognition using DeepFace
        results = DeepFace.find(frame, db_path=db_path, model_name="ArcFace",enforce_detection=False,distance_metric="cosine")
 
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
 
                   
                    return name, x_min, y_min, x_max, y_max
    except Exception as e:
        print(f"Error in face recognition: {e}")
        return None
 
