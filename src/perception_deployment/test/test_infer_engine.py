from ultralytics import YOLO

model = YOLO("src/perception_deployment/perception_deployment/yolo26n.engine")

results = model("src/perception_deployment/images/puppy.png")

results[0].show()
results[0].save("output.png")

print(results)