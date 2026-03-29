from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def detect_vehicles(frame, left_x, right_x):
    results = model(frame, verbose=False)

    illegal_area = 0

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label in ["car", "truck", "bus", "motorcycle"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2

                box_area = (x2 - x1) * (y2 - y1)

                if left_x is not None and right_x is not None:
                    if center_x < left_x or center_x > right_x:
                        color = (0, 0, 255)
                        label_text = "Illegal"
                        illegal_area += box_area   # 👈 ADD THIS (do this later , no need to do now)
                    else:
                        color = (0, 255, 0)
                        label_text = "Vehicle"
                else:
                    color = (0, 255, 0)
                    label_text = "Vehicle"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, illegal_area