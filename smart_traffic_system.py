import cv2
import numpy as np
import time
import csv
import os
from datetime import datetime

# Constants
MIN_WIDTH = 80
MIN_HEIGHT = 80
LINE_OFFSET = 6
DETECTION_LINE_POSITION = 550
FRAME_DELAY = 60

vehicle_centers = []
vehicle_count = 0

# Create directory for snapshots if it doesn't exist
if not os.path.exists('snapshots'):
    os.makedirs('snapshots')

# Create or open the CSV file for logging speed data
csv_file = 'car_speed.csv'
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Vehicle ID', 'Speed (km/h)', 'Timestamp'])

# Function to calculate the center of detected vehicles
def get_center(x, y, w, h):
    center_x = x + int(w / 2)
    center_y = y + int(h / 2)
    return center_x, center_y

# Initialize Video Capture
video = cv2.VideoCapture('v.mp4')

# Background Subtraction Method
background_subtractor = cv2.createBackgroundSubtractorMOG2()

# Start processing frames
while True:
    ret, frame = video.read()
    if not ret:
        break

    time.sleep(1 / FRAME_DELAY)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 5)

    mask = background_subtractor.apply(blurred_frame)
    dilated = cv2.dilate(mask, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    processed_mask = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame, (25, DETECTION_LINE_POSITION), (1200, DETECTION_LINE_POSITION), (255, 127, 0), 3)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = get_center(x, y, w, h)
        vehicle_centers.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

        for (cx, cy) in vehicle_centers[:]:
            if DETECTION_LINE_POSITION - LINE_OFFSET < cy < DETECTION_LINE_POSITION + LINE_OFFSET:
                vehicle_count += 1
                cv2.line(frame, (25, DETECTION_LINE_POSITION), (1200, DETECTION_LINE_POSITION), (0, 127, 255), 3)
                vehicle_centers.remove((cx, cy))

                speed = (w / 100) * 3.6
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([vehicle_count, round(speed, 2), timestamp])

                snapshot_name = f'snapshots/vehicle_{vehicle_count}_{timestamp.replace(":", "-")}.jpg'
                cv2.imwrite(snapshot_name, frame[y:y+h, x:x+w])

                print(f"Vehicle {vehicle_count} detected. Speed: {round(speed, 2)} km/h at {timestamp}")

    cv2.putText(frame, f"VEHICLE COUNT: {vehicle_count}", (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.imshow("Original Video", frame)
    cv2.imshow("Detection Mask", processed_mask)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
video.release()