import cv2
import numpy as np
from plyer import notification
import time

# === CONFIG ===
VIDEO_PATH = "sample_video.mp4"  # Use 0 for webcam
ALERT_TITLE = "Accident Alert!"
ALERT_MESSAGE = "⚠ Accident detected in surveillance feed."
ALERT_SOUND = True

# === YOLO SETUP ===
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

layer_names = net.getUnconnectedOutLayersNames()
cap = cv2.VideoCapture(VIDEO_PATH)

accident_detected = False
safety_message = ""

def get_direction(boxA, boxB):
    # box = (x1, y1, x2, y2)
    # Determine relative position of boxB wrt boxA
    xA_center = (boxA[0] + boxA[2]) / 2
    yA_center = (boxA[1] + boxA[3]) / 2
    xB_center = (boxB[0] + boxB[2]) / 2
    yB_center = (boxB[1] + boxB[3]) / 2

    dx = xB_center - xA_center
    dy = yB_center - yA_center

    if abs(dx) > abs(dy):  # Horizontal dominance
        if dx > 0:
            return "Move Left"
        else:
            return "Move Right"
    else:  # Vertical dominance
        if dy > 0:
            return "Move Up (Forward)"
        else:
            return "Move Down (Backward)"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] in ["car", "truck", "bus", "motorbike"]:
                cx, cy, w, h = (detection[0:4] * [width, height, width, height]).astype("int")
                x, y = int(cx - w / 2), int(cy - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    vehicles = [boxes[i] for i in indexes.flatten()]

    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            x1, y1, w1, h1 = vehicles[i]
            x2, y2, w2, h2 = vehicles[j]
            boxA = (x1, y1, x1 + w1, y1 + h1)
            boxB = (x2, y2, x2 + w2, y2 + h2)

            x_left = max(boxA[0], boxB[0])
            y_top = max(boxA[1], boxB[1])
            x_right = min(boxA[2], boxB[2])
            y_bottom = min(boxA[3], boxB[3])

            if x_right > x_left and y_bottom > y_top:
                iou = (x_right - x_left) * (y_bottom - y_top) / (
                    (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]) +
                    (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]) -
                    (x_right - x_left) * (y_bottom - y_top)
                )
                if iou > 0.3:
                    accident_detected = True

                    # Red boxes on collision vehicles
                    cv2.rectangle(frame, (boxA[0], boxA[1]), (boxA[2], boxA[3]), (0, 0, 255), 2)
                    cv2.rectangle(frame, (boxB[0], boxB[1]), (boxB[2], boxB[3]), (0, 0, 255), 2)

                    class_i = classes[class_ids[i]]
                    class_j = classes[class_ids[j]]

                    cv2.putText(frame, f"Collision: {class_i} & {class_j}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    # Get safety direction relative to vehicle i
                    safety_message = get_direction(boxA, boxB)

    # Draw normal vehicle boxes
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        color = (0, 255, 0)
        label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if accident_detected:
        cv2.putText(frame, ALERT_MESSAGE, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(frame, "Safety: " + safety_message, (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Notification with safety direction
        notification.notify(
            title=ALERT_TITLE,
            message=f"{ALERT_MESSAGE}\nSafety: {safety_message}",
            timeout=10
        )

        if ALERT_SOUND:
            try:
                import winsound
                winsound.Beep(1000, 700)
            except:
                pass

        cv2.imshow("Accident Surveillance", frame)
        cv2.waitKey(5000)  # Show for 5 seconds
        break

    cv2.imshow("Accident Surveillance", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit early
        break

cap.release()
cv2.destroyAllWindows()
