import cv2
import mediapipe as mp
from ultralytics import YOLO

video_path = "recordings/IMG_4206.mov"
output_path = "outputs/ball_pose6.mp4"

cap = cv2.VideoCapture(video_path)

print("Video opened:", cap.isOpened())

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("Original res:", width, "x", height)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (height, width)
)

print("Writer opened:", out.isOpened())

model = YOLO("yolov8n.pt")

BALL_CLASS = 32

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

frame_number = 0

with mp_pose.Pose() as pose:

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.rotate(
            frame,
            cv2.ROTATE_90_CLOCKWISE
        )

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

            left = pose_results.pose_landmarks.landmark[27]
            right = pose_results.pose_landmarks.landmark[28]

            left_x = int(left.x * frame.shape[1])
            left_y = int(left.y * frame.shape[0])

            right_x = int(right.x * frame.shape[1])
            right_y = int(right.y * frame.shape[0])

            cv2.circle(
                frame,
                (left_x, left_y),
                8,
                (255, 0, 0),
                -1
            )

            cv2.circle(
                frame,
                (right_x, right_y),
                8,
                (255, 0, 0),
                -1
            )

        cv2.putText(
            frame,
            f"Time: {t:.2f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        out.write(frame)

        frame_number += 1

        if frame_number % max(1, int(fps)) == 0:
            print("Processed frames:", frame_number)

cap.release()
out.release()

print("Finished!")
print("Frames processed:", frame_number)
print("Output:", output_path)