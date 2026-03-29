import cv2
import numpy as np

def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]

    polygons = np.array([
        [(0, height),
         (width, height),
         (width, int(height * 0.6)),
         (0, int(height * 0.6))]
    ])

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(img, mask)


def make_line_points(y1, y2, line):
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]


def average_slope_intercept(lines):
    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        intercept = y1 - slope * x1

        if slope < -0.5:
            left_fit.append((slope, intercept))
        elif slope > 0.5:
            right_fit.append((slope, intercept))

    left_lane = np.mean(left_fit, axis=0) if left_fit else None
    right_lane = np.mean(right_fit, axis=0) if right_fit else None

    return left_lane, right_lane


def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    cropped = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        cropped,
        2,
        np.pi / 180,
        threshold=100,
        minLineLength=50,
        maxLineGap=10
    )

    left_line = None
    right_line = None

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)

            # Filter strong slopes only
            if abs(slope) < 0.5:
                continue

            if slope < 0:
                left_line = (x1, y1, x2, y2)
            else:
                right_line = (x1, y1, x2, y2)

    left_x = None
    right_x = None

    if left_line is not None:
        x1, y1, x2, y2 = left_line
        left_x = x1
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)

    if right_line is not None:
        x1, y1, x2, y2 = right_line
        right_x = x1
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    return frame, left_x, right_x