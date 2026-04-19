import sys
import os

# Adds the directory containing 'perception_deployment' to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from perception_deployment.trt_bridge import TensorRTInference

import cv2

import time

engine_path = "src/perception_robotics/src/perception_deployment/models/yolo.engine"

trt_eng = TensorRTInference(engine_path)



orig = cv2.imread("src/perception_robotics/src/perception_deployment/images/farp.png")
img_input = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
post_prcessed_data = trt_eng.run(orig)

# # -------------------------
# # Warmup
# # -------------------------
# img, _ = preprocess(orig)
# for _ in range(10):
#     infer(context, bindings, inputs, outputs, img)

# # -------------------------
# # Preprocess
# # -------------------------
# t0 = time.time()
# img, orig = preprocess(orig)
# t1 = time.time()

# # -------------------------
# # Inference
# # -------------------------
# t2 = time.time()
# output = infer(context, bindings, inputs, outputs, img)
# t3 = time.time()

# # -------------------------
# # Postprocess
# # -------------------------
# t4 = time.time()
# filtered = filter_detections(output)
# scaled = scale_boxes(filtered, orig.shape)
# t5 = time.time()

# result = draw_boxes(orig, scaled)

# cv2.imwrite("output_2.jpg", result)

# # -------------------------
# # Print timings
# # -------------------------
# print(f"Preprocess: {(t1 - t0)*1000:.2f} ms")
# print(f"Inference: {(t3 - t2)*1000:.2f} ms")
# print(f"Postprocess: {(t5 - t4)*1000:.2f} ms")
# print(f"Total: {(t5 - t0)*1000:.2f} ms")