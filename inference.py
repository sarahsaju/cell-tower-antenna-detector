from ultralytics import YOLO
import cv2
import csv
import time
model=YOLO(r"runs\detect\train8\weights\best.pt")
video=r"D:\sarah\antennadetector\DJI P3P Video 25   P O I  Cell Phone Tower Next To V V A.mp4"
cap=cv2.VideoCapture(video)
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
orig_fps=cap.get(cv2.CAP_PROP_FPS)
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
slow=2
outvid=cv2.VideoWriter("antennadetected_slowed_video.mp4",fourcc,orig_fps/slow,(width,height))
csv_file=open("boundingboxcoordinates_of_deployedvideo.csv",mode='w',newline="")
csv_writer=csv.writer(csv_file)
csv_writer.writerow(["Frame","Class","Confidence","X1", 'Y1',"X2","Y2"])
frame_no=0
start_time=time.time()
while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        break
    frame_no = frame_no+1
    finals=model(frame,conf=0.3)[0]
    
    for box in finals.boxes:
        cls=int(box.cls[0])
        conf=box.conf[0]
        if model.names[cls]=="antenna":
            x1,y1,x2,y2=map(int,box.xyxy[0])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"Antenna {conf:.2f}",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            csv_writer.writerow([frame_no,"antenna",f"{conf:.2f}",x1,y1,x2,y2])
            
    outvid.write(frame)
cap.release()
end_time=time.time()
total_time=end_time-start_time
fps=frame_no/total_time
print("inference fps",fps)
outvid.release()
            