import cv2
import numpy as np

def preprocess(frame):
    resized = cv2.resize(frame, (640, 640))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    batched = np.expand_dims(chw, axis=0)
    return batched