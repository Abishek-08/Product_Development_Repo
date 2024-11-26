import mediapipe as mp
import cv2

webcam = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face = mp_face_mesh.FaceMesh(max_num_faces=5,static_image_mode = True, min_tracking_confidence= 0.5, min_detection_confidence=0.5)

while True:
    ret,frame = webcam.read()
    
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    result = face.process(rgb_frame)
    #print(dir(result))
    print("MultifaceMark: ",result.multi_face_landmarks)
    
    if result.multi_face_landmarks:
        for i in result.multi_face_landmarks:
            print("Landmark: ",i.landmark[0].x)
            mp_drawing.draw_landmarks(frame,i,mp_face_mesh.FACEMESH_CONTOURS,landmark_drawing_spec= mp_drawing.DrawingSpec(circle_radius=1,color=(0,255,0),thickness=2))
    
    
    cv2.imshow("webcam",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break