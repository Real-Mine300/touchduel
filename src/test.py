import os
import cv2
import mediapipe as mp
from ultralytics import YOLO
import csv
import numpy as np


video_path = "recordings/Rain_3.mov"
stem = os.path.splitext(os.path.basename(video_path))[0]
output_path = f"outputs/test_{stem}.png"
cap = cv2.VideoCapture(video_path)
proximity_file = open(f"outputs/proximity_{stem}.csv", "r")

print("Video opened:", cap.isOpened())

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("Original res:", width, "x", height)

roll = int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) % 360
already_upright = cap.get(cv2.CAP_PROP_ORIENTATION_AUTO) == 1

quarter_turn = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE
}

if not already_upright and roll in (90, 270):
    width, height = height, width

print(quarter_turn)
print(roll)
print(already_upright)
print(width)
print(height)

sucess, frame = cap.read()

if not already_upright and roll in (90, 270):
    frame = cv2.rotate(frame, quarter_turn[roll])

cv2.imwrite(output_path, frame)

frame_number = 0

while frame_number < 7:
    sucess, frame = cap.read()

    print(cap.get(cv2.CAP_PROP_POS_MSEC))

    frame_number += 1