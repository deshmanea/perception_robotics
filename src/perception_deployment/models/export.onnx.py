from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.export(
    format="onnx",
    opset=13,     
    dynamic=False,   
    simplify=True
)