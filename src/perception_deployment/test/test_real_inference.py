from perception_deployment.trt.engine import load_engine
from perception_deployment.trt.preprocess import preprocess
from perception_deployment.trt.infer import allocate_buffers, infer
from perception_deployment.trt.postprocess import filter_detections, scale_boxes
from perception_deployment.utils.draw import draw_boxes
import cv2

import time

engine, context = load_engine("/perception_deployment/yolo.engine")
inputs, outputs, bindings = allocate_buffers(engine)

orig = cv2.imread("/workspace/src/perception_deployment/images/puppy.png")

# -------------------------
# Warmup
# -------------------------
img, _ = preprocess(orig)
for _ in range(10):
    infer(context, bindings, inputs, outputs, img)

# -------------------------
# Preprocess
# -------------------------
t0 = time.time()
img, orig = preprocess(orig)
t1 = time.time()

# -------------------------
# Inference
# -------------------------
t2 = time.time()
output = infer(context, bindings, inputs, outputs, img)
t3 = time.time()

# -------------------------
# Postprocess
# -------------------------
t4 = time.time()
filtered = filter_detections(output)
scaled = scale_boxes(filtered, orig.shape)
t5 = time.time()

result = draw_boxes(orig, scaled)

cv2.imwrite("output_2.jpg", result)

# -------------------------
# Print timings
# -------------------------
print(f"Preprocess: {(t1 - t0)*1000:.2f} ms")
print(f"Inference: {(t3 - t2)*1000:.2f} ms")
print(f"Postprocess: {(t5 - t4)*1000:.2f} ms")
print(f"Total: {(t5 - t0)*1000:.2f} ms")