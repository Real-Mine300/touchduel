import cv2
import mediapipe as mp

video_path = "recordings/IMG_4207.mov"
output_path = "outputs/pose_overlay.mp4"

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

        else:

            all_landmarks.append(None)

        out.write(frame)

        frame_number += 1

        if frame_number % int(fps) == 0:
            print("Processed frames:", frame_number)

cap.release()
out.release()

print("Finished!")
print("Frames processed:", frame_number)
print("Landmark frames stored:", len(all_landmarks))
print("Output:", output_path)