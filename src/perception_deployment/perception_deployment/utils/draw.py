import cv2

def draw_boxes(image, detections):
    for det in detections:
        x1, y1, x2, y2, conf, cls = det

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{int(cls)}: {conf:.2f}"

        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image