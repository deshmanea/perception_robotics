from ultralytics import YOLO

model = YOLO("../yolo26n.pt")

# export TensorRT engine
model.export(format="engine", device=0)

print("TensorRT engine created")


model = YOLO("src/perception_deployment/perception_deployment/yolo26n.engine")

results = model("src/perception_deployment/images/puppy.png")
results[0].show()
results[0].save("output.png")

print(results)