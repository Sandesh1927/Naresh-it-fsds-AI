from ultralytics import YOLO
model = YOLO('yolov8n.pt')
import numpy 


detection_output= model.predict (source=r"C:\Users\sande\OneDrive\Desktop\village.jpg", conf=0.25, save=True ) 
                                

print(detection_output)