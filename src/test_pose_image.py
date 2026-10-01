import os
import cv2
import mediapipe as mp

video_path = "recordings/Rain_2.MOV"
stem = os.path.splitext(os.path.basename(video_path))[0]
output_path = f"outputs/first_frame_pose_{stem}.jpg"

cap = cv2.VideoCapture(video_path)

success, image = cap.read()

print("Frame loaded:", success)

cap.release()

if not success:
    print("Could not read first frame.")
    exit()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with mp_pose.Pose() as pose:

    results = pose.process(rgb_image)

    print("Pose detected:", results.pose_landmarks is not None)

    if results.pose_landmarks:

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

cv2.imwrite(output_path, image)

print("Saved:", output_path)