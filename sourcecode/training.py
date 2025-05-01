from ultralytics import YOLO
model=YOLO('yolov8s.yaml')
results=model.train(data="config.yaml",epochs=100,profile=True)