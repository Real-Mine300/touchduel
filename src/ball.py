import os
import cv2
from ultralytics import YOLO
import csv
import matplotlib.pyplot as plt


video_path = "recordings/Rain_1.MOV"
stem = os.path.splitext(os.path.basename(video_path))[0]
output_path = f"outputs/ball_overlay_{stem}.mp4"
csv_path = f"outputs/ball_{stem}.csv"
plot_path = f"outputs/ball_y_{stem}.png"

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

ball_file = open(csv_path, "w", newline="")
writer = csv.writer(ball_file)
writer.writerow(["t", "ball_x", "ball_y", "size_x", "size_y", "conf"])

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(output_path, fourcc, fps, (height, width) )
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

        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 4)
        cv2.circle(frame, (int(center[0]), int(center[1])), 8, (0, 255, 0), -1)

        print(ball_pos)
    else:
        writer.writerow([t,"", "", "", "", ""])

    out.write(frame)

    frame_number += 1
    print(t)


ball_file.close()

print("Output: ", output_path)

ts = []
ys = []

with open(csv_path) as f:
    for row in csv.DictReader(f):
        ts.append(float(row["t"]))
        if row["ball_y"] == "":
            ys.append(float("nan"))
        else:
            ys.append(float(row["ball_y"]))

out.release()


plt.plot(ts, ys)
plt.xlabel("t (s)")
plt.ylabel("ball y (px)")
plt.savefig(plot_path)
plt.show()
