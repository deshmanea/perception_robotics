import numpy as np

def filter_detections(output, conf_threshold=0.5):
    detections = output[0]
    return detections[detections[:, 4] > conf_threshold]


def scale_boxes(detections, orig_shape, input_size=(640, 640)):
    orig_h, orig_w = orig_shape[:2]
    scale_x = orig_w / input_size[0]
    scale_y = orig_h / input_size[1]

    scaled = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det

        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y

        scaled.append([x1, y1, x2, y2, conf, cls])

    return np.array(scaled)