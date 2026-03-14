from ultralytics import YOLO

# Load YOLO model
model = YOLO("../yolo26n.pt")

# Export to ONNX
model.export(format="onnx", opset=12)

print("ONNX export completed")