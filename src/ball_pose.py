import os
import cv2
import mediapipe as mp
from sympy import true
from ultralytics import YOLO
import csv

rotateIMG = true

video_path = "recordings/Rain_3.mov"
stem = os.path.splitext(os.path.basename(video_path))[0]
output_path = f"outputs/ball_pose_{stem}.mp4"
cap = cv2.VideoCapture(video_path)
proximity_file = open(f"outputs/proximity_{stem}.csv", "w")

print("Video opened:", cap.isOpened())

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("Original res:", width, "x", height)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

if rotateIMG:
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (height, width)
    )
else:
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

print("Writer opened:", out.isOpened())

model = YOLO("yolov8n.pt")

BALL_CLASS = 32

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

frame_number = 0


writer = csv.writer(proximity_file)
writer.writerow(["t", "instep_l_x", "instep_l_y", "instep_r_x", "instep_r_y", "knee_l_x", "knee_l_y", "knee_r_x", "knee_r_y", "ankle_l_x", "ankle_l_y", "ankle_r_x", "ankle_r_y", "ball_x", "ball_y", "size_x", "size_y", "conf", "proximity_instep", "proximity_knee", "touch"])

with mp_pose.Pose() as pose:

    while True:

        success, frame = cap.read()

        if not success:
            break

        if rotateIMG: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        t = frame_number / fps

        results = model(frame, verbose=False)

        ball_found = False

        if len(results) > 0 and results[0].boxes is not None:

            boxes = results[0].boxes

            ball_indices = []

            for i, cls in enumerate(boxes.cls):
                if int(cls) == BALL_CLASS:
                    ball_indices.append(i)

            if len(ball_indices) > 0:

                best_index = max(
                    ball_indices,
                    key=lambda i: float(boxes.conf[i])
                )

                left, top, right, bottom = (
                    boxes.xyxy[best_index].tolist()
                )

                center_x = (left + right) / 2
                center_y = (top + bottom) / 2

                cv2.rectangle(
                    frame,
                    (int(left), int(top)),
                    (int(right), int(bottom)),
                    (0, 255, 0),
                    4
                )

                cv2.circle(
                    frame,
                    (int(center_x), int(center_y)),
                    8,
                    (0, 255, 0),
                    -1
                )

                ball_found = True
            else:
                ball_found = False
                center_x = None
                center_y = None
                boxes = None
                best_index = None
                left = None
                top = None
                right = None
                bottom = None

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        pose_results = pose.process(rgb_frame)

        if pose_results.pose_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            ankle_left = pose_results.pose_landmarks.landmark[27]
            ankle_right = pose_results.pose_landmarks.landmark[28]

            ankle_left_x = int(ankle_left.x * frame.shape[1])
            ankle_left_y = int(ankle_left.y * frame.shape[0])

            ankle_right_x = int(ankle_right.x * frame.shape[1])
            ankle_right_y = int(ankle_right.y * frame.shape[0])

            toe_left = pose_results.pose_landmarks.landmark[31]
            toe_right = pose_results.pose_landmarks.landmark[32]

            toe_left_x = int(toe_left.x * frame.shape[1])
            toe_left_y = int(toe_left.y * frame.shape[0])

            toe_right_x = int(toe_right.x * frame.shape[1])
            toe_right_y = int(toe_right.y * frame.shape[0])

            knee_left = pose_results.pose_landmarks.landmark[25]
            knee_right = pose_results.pose_landmarks.landmark[26]

            knee_left_x = int(knee_left.x * frame.shape[1])
            knee_left_y = int(knee_left.y * frame.shape[0])

            knee_right_x = int(knee_right.x * frame.shape[1])
            knee_right_y = int(knee_right.y * frame.shape[0])

            instep_left_x = ankle_left_x + (toe_left_x - ankle_left_x) * 0.6
            instep_left_y = ankle_left_y + (toe_left_y - ankle_left_y) * 0.6
            instep_right_x = ankle_right_x + (toe_right_x - ankle_right_x) * 0.6
            instep_right_y = ankle_right_y + (toe_right_y - ankle_right_y) * 0.6

            cv2.circle(
                frame,
                (toe_left_x, toe_left_y),
                8,
                (255, 0, 0),
                -1
            )
            cv2.circle(
                frame,
                (toe_right_x, toe_right_y),
                8,
                (255, 0, 0),
                -1
            )

            cv2.circle(
                frame,
                (ankle_left_x, ankle_left_y),
                8,
                (255, 100, 0),
                -1
            )

            cv2.circle(
                frame,
                (ankle_right_x, ankle_right_y),
                8,
                (255, 100, 0),
                -1
            )

            cv2.circle(
                frame,
                (knee_left_x, knee_left_y),
                8,
                (100, 100, 200),
                -1
            )

            cv2.circle(
                frame,
                (knee_right_x, knee_right_y),
                8,
                (100, 100, 200),
                -1
            )

            cv2.circle(
                frame,
                (int(instep_left_x), int(instep_left_y)),
                8,
                (157, 255, 255),
                -1
            )

            cv2.circle( 
                frame,
                (int(instep_right_x), int(instep_right_y)),
                8,
                (157, 255, 255),
                -1
            )
        else:
            instep_left_x = None
            instep_left_y = None
            instep_right_x = None
            instep_right_y = None
            ankle_left_x = None
            ankle_left_y = None
            ankle_right_x = None
            ankle_right_y = None
            knee_left_x = None
            knee_left_y = None
            knee_right_x = None
            knee_right_y = None

        cv2.putText(
            frame,
            f"Time: {t:.2f}s",
            (200, 900),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        if ball_found and pose_results.pose_landmarks:
            proximity_right_x = abs(instep_right_x - center_x)
            proximity_right_y = abs(instep_right_y - center_y)
            proximity_left_x = abs(instep_left_x - center_x)
            proximity_left_y = abs(instep_left_y - center_y)
            size_y = bottom - top
            size_x = right - left
            touch_right = proximity_right_x <= size_x and proximity_right_y <= size_y
            touch_left = proximity_left_x <= size_x and proximity_left_y <= size_y
            touch = touch_right or touch_left
            proximity_knee_right_x = abs(knee_right_x - center_x)
            proximity_knee_right_y = abs(knee_right_y - center_y)
            proximity_knee_left_x = abs(knee_left_x - center_x)
            proximity_knee_left_y = abs(knee_left_y - center_y)

        else:
            proximity = False
            touch = False

        if ball_found and pose_results.pose_landmarks:
            proximity = min(
                (proximity_left_x ** 2 + proximity_left_y ** 2) ** 0.5,
                (proximity_right_x ** 2 + proximity_right_y ** 2) ** 0.5,
            )
            proximity_knee = min(
                (proximity_knee_left_x ** 2 + proximity_knee_left_y ** 2) ** 0.5,
                (proximity_knee_right_x ** 2 + proximity_knee_right_y ** 2) ** 0.5,
            )
            writer.writerow([
                t, instep_left_x, instep_left_y, instep_right_x, instep_right_y,
                knee_left_x, knee_left_y, knee_right_x, knee_right_y,
                ankle_left_x, ankle_left_y, ankle_right_x, ankle_right_y,
                center_x, center_y, size_x, size_y, float(boxes.conf[best_index]),
                proximity, proximity_knee, touch,
            ])
        elif not ball_found and pose_results.pose_landmarks:
            writer.writerow([t, instep_left_x, instep_left_y, instep_right_x,
            instep_right_y, knee_left_x, knee_left_y, knee_right_x, knee_right_y,
            ankle_left_x, ankle_left_y, ankle_right_x, ankle_right_y, 
            None, None, None, None, None, None, None, None, None, None])
        elif ball_found and not pose_results.pose_landmarks:
            writer.writerow([t, None, None, None, None, None, None, None, None, None, None, None, None, center_x, center_y, size_x, size_y, float(boxes.conf[best_index]), None, None, None])
        else:
            writer.writerow([t, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None])
        
        
        out.write(frame)

        frame_number += 1

        if frame_number % max(1, int(fps)) == 0:
            print("Processed frames:", frame_number)

cap.release()
out.release()

print("Finished!")
print("Frames processed:", frame_number)
print("Output:", output_path)
print("Proximity file:", proximity_file.name)