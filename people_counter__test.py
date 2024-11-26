import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*

model=YOLO('yolov8n.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('cutv1.mp4')


count=0

tracker=Tracker()

ly1min=278
ly1max=343
ly2min = 219
ly2max = 270

offset=6
entry_count = set()
exit_count = set()
entry_count_dic = {}
exit_count_dic = {}

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(860, 480))
   

    results=model.predict(frame,classes=[0])
 
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    print(px)
    list=[]
             
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
       
        
        list.append([x1,y1,x2,y2])
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        
        print("cyyy: ",cy," ID: ",id)
        print(entry_count)
        if cy > ly2min and cy < ly2max:
            entry_count_dic[id]=cy
        if id in entry_count_dic:
           if cy > ly1min and cy < ly1max:
              cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
              cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
              entry_count.add(id)
              print("Count: ",len(entry_count))
              print("Entry Dictiory: ",entry_count_dic)
              
        if cy > ly1min and cy < ly1max:
            exit_count_dic[id]=cy
        if id in exit_count_dic:
           if cy > ly2min and cy < ly2max:
              cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
              cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
              exit_count.add(id)
              
           


    cv2.line(frame,(509,ly1min),(675,ly1max),(0,255,0),2)
    cv2.line(frame,(580,ly2min),(700,ly2max),(0,0,255),2)
    cv2.putText(frame,"Entry-count: "+str(len(entry_count)),(20,100),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
    cv2.putText(frame,"Exit-count: "+str(len(exit_count)),(20,140),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
    cv2.putText(frame,"Total-count: "+str(len(entry_count)-len(exit_count)),(20,180),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
