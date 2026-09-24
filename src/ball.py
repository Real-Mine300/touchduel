import cv2
from ultralytics import YOLO
import csv
import matplotlib.pyplot as plt


video_path = "recordings/IMG_4209.mov"
output_path = "outputs/ball.csv"

cap = cv2.VideoCapture(video_path)

print("Video opened:", cap.isOpened())

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("res:", width, "x", height)

set_time = 0
cap.set(cv2.CAP_PROP_POS_MSEC, set_time)

all_ball_pos = []

model = YOLO("yolov8n.pt")

frame_number = 0

ball_file = open("outputs/ball.csv", "w", newline="")
writer = csv.writer(ball_file)
writer.writerow(["t", "ball_x", "ball_y", "size_x", "size_y", "conf"])

while True:

    success, frame = cap.read()

    if not success:
        break


    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    results = model(frame)
    t = frame_number/fps
    classes = [int(c) for c in results[0].boxes.cls]
    if 32 in classes:
        i = classes.index(32)

        confidence = float(results[0].boxes.conf[i])
        left, top, right, bottom = results[0].boxes.xyxy[i].tolist()

        print(confidence)
        print(left, top, right, bottom)

        center = [(left+right)/2, (top+bottom)/2]
        print(center)

        ball_pos = {
            "x": center[0],
            "y": center[1],
            "size_x": abs(right-left),
            "size_y": abs(top-bottom),
            "confidence":confidence
        }

        all_ball_pos.append(ball_pos)
        writer.writerow([t,
            center[0],
            center[1],
            abs(right-left),
            abs(top-bottom),
            confidence
        ])

        print(ball_pos)
    else:
        writer.writerow([t,"", "", "", "", ""])

    frame_number += 1
    print(t)

ball_file.close()

print("Output: ", output_path)

ts = []
ys = []

with open("outputs/ball.csv") as f:
    for row in csv.DictReader(f):
        ts.append(float(row["t"]))
        if row["ball_y"] == "":
            ys.append(float("nan"))
        else:
            ys.append(float(row["ball_y"]))


plt.plot(ts, ys)
plt.xlabel("t (s)")
plt.xlabel("ball y (px)")
plt.savefig("outputs/ball_y.png")
plt.show()