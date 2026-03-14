from ultralytics import YOLO

model = YOLO("../yolo26n.pt")

# export TensorRT engine
model.export(format="engine", device=0)

print("TensorRT engine created")
