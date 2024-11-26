import cv2

rtsp_url = "rtsp://admin:webcam123@192.168.1.64/Streaming/channels/101"
cap = cv2.VideoCapture(rtsp_url)

while True:
    ret,frame = cap.read()
    
    cv2.imshow("webcam",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

