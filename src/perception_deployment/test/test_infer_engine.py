from ultralytics import YOLO

model = YOLO("src/perception_robotics/src/perception_deployment/models/yolo.engine")

results = model("src/perception_robotics/src/perception_deployment/images/puppy.png")

results[0].show()
results[0].save("output.png")

print(results)