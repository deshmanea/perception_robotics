import cv2
import numpy as np

def preprocess(img, input_size=(640, 640)):
    orig = img.copy()

    img = cv2.resize(img, input_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    img = np.ascontiguousarray(img)

    return img, orig