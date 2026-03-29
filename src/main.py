import cv2
import numpy as np
from lane_detection import detect_lanes
from vehicle_detection import detect_vehicles
import time

prev_time = 0
prev_occupancy = 0
alpha = 0.3  # smoothing factor for occupancy calculation (tried 0.6 but was too harsh)
cap = cv2.VideoCapture("data/road2.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    #Lane Detection (abhi ke liye simple)
    lane_frame, left_x, right_x = detect_lanes(frame)
    lane_area = 0

    if left_x is not None and right_x is not None:
        y1 = height
        y2 = int(height * 0.6)

        bottom_width = abs(right_x - left_x)
        top_width = int(bottom_width * 0.6)  # approximate narrowing

        lane_height = y1 - y2

        # trapezium area formula
        lane_area = (bottom_width + top_width) / 2 * lane_height

    #Legal Zone Overlay (clean + subtle)
    if left_x is not None and right_x is not None:
        overlay = lane_frame.copy()

        points = np.array([[
            (left_x, height),
            (right_x, height),
            (right_x, int(height * 0.6)),
            (left_x, int(height * 0.6))
        ]])

        cv2.fillPoly(overlay, points, (0, 200, 0))
        lane_frame = cv2.addWeighted(overlay, 0.12, lane_frame, 0.88, 0)

        # Lane Center and Vehicle Center Logic (corrected)
        lane_center = (left_x + right_x) // 2
        frame_center = width // 2
        offset = frame_center - lane_center

        # Draw lane center (thin)
        cv2.line(lane_frame,
                 (lane_center, height),
                 (lane_center, int(height * 0.6)),
                 (255, 255, 0), 2)

        # Draw vehicle center
        cv2.line(lane_frame,
                 (frame_center, height),
                 (frame_center, int(height * 0.6)),
                 (0, 255, 255), 2)

        # Offset logic with thresholding for better readability
        threshold = 20

        if abs(offset) < threshold:
            status = "Centered"
        elif offset > 0:
            status = "Drifting Right"
        else:
            status = "Drifting Left"



    # Vehicle Detection (ONLY ONCE)
    final_frame, illegal_area = detect_vehicles(lane_frame, left_x, right_x)
    if lane_area > 0:
        occupancy = (illegal_area / lane_area) * 100
        occupancy = min(occupancy, 100)
    else:
        occupancy = 0

    # remove tiny noise
    if occupancy < 2:
        occupancy = 0

    # limit sudden jumps
    max_change = 5
    if abs(occupancy - prev_occupancy) > max_change:
        if occupancy > prev_occupancy:
            occupancy = prev_occupancy + max_change
        else:
            occupancy = prev_occupancy - max_change

    # smoothing
    alpha = 0.3
    occupancy = alpha * occupancy + (1 - alpha) * prev_occupancy

    prev_occupancy = occupancy

    # FPS Calculation (will tune after tuesday progress report)
    current_time = time.time()
    fps = 1 / (current_time - prev_time + 1e-6)
    prev_time = current_time

    if left_x is None or right_x is None:
        offset = 0
        status = "N/A"

    cv2.putText(final_frame,
            "Lane Detection | Occupancy Analysis",
            (width // 3, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1)
    

    # Create semi-transparent panel
    overlay_ui = final_frame.copy()

    cv2.rectangle(overlay_ui, (10, 10), (350, 200), (0, 0, 0), -1)
    final_frame = cv2.addWeighted(overlay_ui, 0.6, final_frame, 0.4, 0)
    
    # ===== UI DISPLAY =====

    y = 40
    gap = 40

    # FPS
    cv2.putText(final_frame, f"FPS: {int(fps)}",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Offset already calculated earlier → just display again cleanly
    y += gap
    cv2.putText(final_frame,
                f"Lane Offset: {abs(offset)} px | {status}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2)

    # Occupancy color logic
    y += gap
    if occupancy < 25:
        occ_color = (0, 255, 0)
    elif occupancy < 50:
        occ_color = (0, 255, 255)
    else:
        occ_color = (0, 0, 255)

    cv2.putText(final_frame,
                f"Road Occupancy: {occupancy:.1f}%",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                occ_color,
                2)

    # Status
    y += gap
    if occupancy < 25:
        status_text = "Free Flow"
    elif occupancy < 49:
        status_text = "Moderate Traffic"
    else:
        status_text = "High Obstruction"


    cv2.putText(final_frame,
                f"Status: {status_text}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                occ_color,
                2)

    # Illegal count (no need as i skipped it for now, but can be added back easily)
    y += gap
    
    cv2.imshow("Advanced Vision System (PE1)", final_frame)
    out.write(final_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()