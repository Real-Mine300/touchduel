import os
import cv2
import mediapipe as mp
import csv


video_path = "recordings/Rain_2.MOV"
stem = os.path.splitext(os.path.basename(video_path))[0]
output_path = f"outputs/pose_overlay_{stem}.mp4"
csv_path = f"outputs/ankles_{stem}.csv"

cap = cv2.VideoCapture(video_path)

print("Video opened:", cap.isOpened())

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("res:", width, "x", height)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (height, width)
)

print("Writer opened:", out.isOpened())

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

all_landmarks = []

ankle_file = open(csv_path, "w", newline="")
writer = csv.writer(ankle_file)
writer.writerow(["t", "ankle_l_x", "ankle_l_y", "ankle_r_x", "ankle_r_y"])

with mp_pose.Pose() as pose:

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.rotate(
            frame,
            cv2.ROTATE_90_CLOCKWISE
        )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = pose.process(rgb_frame)

        if results.pose_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            frame_landmarks = {}

            for landmark_number, landmark in enumerate(
                results.pose_landmarks.landmark
            ):

                frame_landmarks[landmark_number] = {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                }

            all_landmarks.append(frame_landmarks)
            t = frame_number / fps
            left = frame_landmarks[27]
            right = frame_landmarks[28]
            writer.writerow([
                t,
                left["x"] * frame.shape[1],
                left["y"] * frame.shape[0],
                right["x"] * frame.shape[1],
                right["y"] * frame.shape[0],
            ])

        else:

            all_landmarks.append(None)
            writer.writerow([frame_number / fps, "", "", "", ""])

        out.write(frame)

        frame_number += 1

        if frame_number % int(fps) == 0:
            print("Processed frames:", frame_number)

cap.release()
out.release()
ankle_file.close()

print("Finished!")
print("Frames processed:", frame_number)
print("Landmark frames stored:", len(all_landmarks))
print("Output:", output_path)
