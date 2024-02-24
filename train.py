from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  
model = YOLO("yolov8n.pt")


model.train(data="coco128.yaml", epochs=3)  
metrics = model.val()  
path = model.export(format="onnx")  