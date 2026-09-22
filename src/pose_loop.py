import cv2
import mediapipe as mp

video_path = "recordings/VId.mov"

cap = cv2.VideoCapture(video_path)

print("Video openeded:", cap.isOpened())

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("res:", width, "x", height)

mp_pose = mp.solutions.pose

all_landmarks = []

with mp_pose.Pose() as pose:
    frame_number = 0

    while True:
        sucess, frame = cap.read()

        if not sucess:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.Color_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.poselandmarks:
            frame_landmarks = {}

            for landmark_number, landmark in enumerate(results.pose_landmarks.landmark):
                frame_landmarks[landmark_number] = {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                }

            all_landmarks.append(frame_landmarks)
        else:
            all_landmarks.append(None)

        frame_number +=1

cap.realease()
print("Finished!")
print("Frames processed: ", frame_number)
print("Landmark frames stored: ", len(all_landmarks))