import os
import cv2
import mediapipe as mp
from ultralytics import YOLO
import csv
import numpy as np


video_path = "recordings/Rain_3.mov"
stem = os.path.splitext(os.path.basename(video_path))[0]
output_path = f"outputs/touch_pose_{stem}.mp4"
cap = cv2.VideoCapture(video_path)
proximity_file = open(f"outputs/proximity_{stem}.csv", "w")

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
    90: cv2.ROTATE_180,
    90: cv2.ROTATE_90_COUNTERCLOCKWISE
}

if not already_upright and roll in (90, 270):
    width, height = height, width

model = YOLO("yolov8n.pt")

BALL_CLASS = 32

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

frame_number = 0


writer = csv.writer(proximity_file)
writer.writerow(["t", "instep_l_x", "instep_l_y", "instep_r_x", "instep_r_y", "knee_l_x", "knee_l_y", "knee_r_x", "knee_r_y", "ankle_l_x", "ankle_l_y", "ankle_r_x", "ankle_r_y", "wrist_l_x", "wrist_l_y", "wrist_r_x", "wrist_r_y", "ball_x", "ball_y", "size_x", "size_y", "conf", "proximity_instep", "proximity_knee", "proximity_wrist", "touch"])

prev_gray = None
prev_pts = None
cam_dx_list = []
cam_dy_list = []
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)

with mp_pose.Pose() as pose:

    while True:

        success, frame = cap.read()

        if not success:
            break

        if not already_upright and roll in quarter_turn:
            frame = cv2.rotate(frame, quarter_turn[roll])
        t = frame_number / fps

        results = model(frame, verbose=False)

        ball_found = False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cam_dx, cam_dy, cam_source = np.nan, np.nan, ""

        if prev_gray is not None and prev_pts is not None and len(prev_pts) >= 8:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
            good = status.reshape(-1) == 1
            if good.sum() >= 8:
                flow = (nxt - prev_pts).reshape(-1, 2)[good]
                med = np.median(flow, axis=0)
                mad = np.median(np.abs(flow - med), axis=0) + 1e-6
                inliers = np.all(np.abs(flow - med) < 3 * mad, axis=1)
                if inliers.sum() >= 8:
                    cam_dx, cam_dy = np.median(flow[inliers], axis=0)
                    cam_source = "bg"

        mask = np.full(gray.shape, 255, np.uint8)
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=80, qualityLevel=0.01, minDistance=24, mask=mask)
        prev_pts = pts if pts is not None else prev_pts
        prev_gray = gray


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

            wrist_left = pose_results.pose_landmarks.landmark[15]
            wrist_right = pose_results.pose_landmarks.landmark[16]

            wrist_left_x = int(wrist_left.x * frame.shape[1])
            wrist_left_y = int(wrist_left.y * frame.shape[0])

            wrist_right_x = int(wrist_right.x * frame.shape[1])
            wrist_right_y = int(wrist_right.y * frame.shape[0])

            instep_left_x = ankle_left_x + (toe_left_x - ankle_left_x) * 0.6
            instep_left_y = ankle_left_y + (toe_left_y - ankle_left_y) * 0.6
            instep_right_x = ankle_right_x + (toe_right_x - ankle_right_x) * 0.6
            instep_right_y = ankle_right_y + (toe_right_y - ankle_right_y) * 0.6

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
            wrist_left_x = None
            wrist_left_y = None
            wrist_right_x = None
            wrist_right_y = None

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
            proximity_wrist_left_x = abs(wrist_left_x - center_x)
            proximity_wrist_left_y = abs(wrist_left_y - center_y)
            proximity_wrist_right_x = abs(wrist_right_x - center_x)
            proximity_wrist_right_y = abs(wrist_right_y - center_y)
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
            proximity_wrist = min(
                (proximity_wrist_left_x ** 2 + proximity_wrist_left_y ** 2) ** 0.5,
                (proximity_wrist_right_x ** 2 + proximity_wrist_right_y ** 2) ** 0.5,
            )
            writer.writerow([
                t, instep_left_x, instep_left_y, instep_right_x, instep_right_y,
                knee_left_x, knee_left_y, knee_right_x, knee_right_y,
                ankle_left_x, ankle_left_y, ankle_right_x, ankle_right_y,
                wrist_left_x, wrist_left_y, wrist_right_x, wrist_right_y,
                center_x, center_y, size_x, size_y, float(boxes.conf[best_index]),
                proximity, proximity_knee, proximity_wrist, touch,
            ])
        elif not ball_found and pose_results.pose_landmarks:
            writer.writerow([t, instep_left_x, instep_left_y, instep_right_x,
            instep_right_y, knee_left_x, knee_left_y, knee_right_x, knee_right_y,
            ankle_left_x, ankle_left_y, ankle_right_x, ankle_right_y, 
            wrist_left_x, wrist_left_y, wrist_right_x, wrist_right_y,
            None, None, None, None, None, None, None])
        elif ball_found and not pose_results.pose_landmarks:
            writer.writerow([t, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, center_x, center_y, size_x, size_y, float(boxes.conf[best_index]), None, None, None, None])
        else:
            writer.writerow([t, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None])
        
        

        frame_number += 1

        if frame_number % max(1, int(fps)) == 0:
            print("Processed frames:", frame_number)

cap.release()

print("Finished!")
print("Frames processed:", frame_number)
print("Output:", output_path)
print("Proximity file:", proximity_file.name)

with open(proximity_file.name, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    data = {header: [] for header in reader.fieldnames}
    for row in reader:
        for header in reader.fieldnames:
            data[header].append(row[header])


t = np.array(data['t'])
instep_left_x = np.array(data['instep_l_x'])
instep_left_y = np.array(data['instep_l_y'])
instep_right_x = np.array(data['instep_r_x'])
instep_right_y = np.array(data['instep_r_y'])
knee_left_x = np.array(data['knee_l_x'])
knee_left_y = np.array(data['knee_l_y'])
knee_right_x = np.array(data['knee_r_x'])
knee_right_y = np.array(data['knee_r_y'])
ankle_left_x = np.array(data['ankle_l_x'])
ankle_left_y = np.array(data['ankle_l_y'])
ankle_right_x = np.array(data['ankle_r_x'])
ankle_right_y = np.array(data['ankle_r_y'])
wrist_left_x = np.array(data['wrist_l_x'])
wrist_left_y = np.array(data['wrist_l_y'])
wrist_right_x = np.array(data['wrist_r_x'])
wrist_right_y = np.array(data['wrist_r_y'])
ball_x = np.array(data['ball_x'])
ball_y = np.array(data['ball_y'])
size_x = np.array(data['size_x'])
size_y = np.array(data['size_y'])
conf = np.array(data['conf'])
cam_dx = np.array(cam_dx_list, dtype=float)
cam_dy = np.array(cam_dy_list, dtype=float)
step_x = np.nan_to_num(cam_dx, nan=0.0)
step_y = np.nan_to_num(cam_dy, nan=0.0)
cam_pos_x = np.cumsum(step_x)
cam_pos_y = np.cumsum(step_y)
proximity_instep = np.array(data['proximity_instep'])
proximity_knee = np.array(data['proximity_knee'])
proximity_wrist = np.array(data['proximity_wrist'])
touch = np.array(data['touch'])

t = np.array([float(v) for v in data["t"]], dtype=float)

#clean
ball_x -= cam_pos_x
ball_y -= cam_pos_y
instep_left_x -= cam_pos_x
instep_left_y -= cam_pos_y
instep_right_x -= cam_pos_x
instep_right_y -= cam_pos_y
knee_left_x -= cam_pos_x
knee_left_y -= cam_pos_y
knee_right_x -= cam_pos_x
knee_right_y -= cam_pos_y
ankle_left_x -= cam_pos_x
ankle_left_y -= cam_pos_y
ankle_right_x -= cam_pos_x
ankle_right_y -= cam_pos_y
wrist_left_x -= cam_pos_x
wrist_left_y -= cam_pos_y
wrist_right_x -= cam_pos_x
wrist_right_y -= cam_pos_y

# acccount for "" into nan
instep_left_x = np.array([np.nan if v in ("", None) else float(v) for v in data["instep_l_x"]], dtype=float)
instep_left_y = np.array([np.nan if v in ("", None) else float(v) for v in data["instep_l_y"]], dtype=float)
instep_right_x = np.array([np.nan if v in ("", None) else float(v) for v in data["instep_r_x"]], dtype=float)
instep_right_y = np.array([np.nan if v in ("", None) else float(v) for v in data["instep_r_y"]], dtype=float)
knee_left_x = np.array([np.nan if v in ("", None) else float(v) for v in data["knee_l_x"]], dtype=float)
knee_left_y = np.array([np.nan if v in ("", None) else float(v) for v in data["knee_l_y"]], dtype=float)
knee_right_x = np.array([np.nan if v in ("", None) else float(v) for v in data["knee_r_x"]], dtype=float)
knee_right_y = np.array([np.nan if v in ("", None) else float(v) for v in data["knee_r_y"]], dtype=float)
ankle_left_x = np.array([np.nan if v in ("", None) else float(v) for v in data["ankle_l_x"]], dtype=float)
ankle_left_y = np.array([np.nan if v in ("", None) else float(v) for v in data["ankle_l_y"]], dtype=float)
ankle_right_x = np.array([np.nan if v in ("", None) else float(v) for v in data["ankle_r_x"]], dtype=float)
ankle_right_y = np.array([np.nan if v in ("", None) else float(v) for v in data["ankle_r_y"]], dtype=float)
wrist_left_x = np.array([np.nan if v in ("", None) else float(v) for v in data["wrist_l_x"]], dtype=float)
wrist_left_y = np.array([np.nan if v in ("", None) else float(v) for v in data["wrist_l_y"]], dtype=float)
wrist_right_x = np.array([np.nan if v in ("", None) else float(v) for v in data["wrist_r_x"]], dtype=float)
wrist_right_y = np.array([np.nan if v in ("", None) else float(v) for v in data["wrist_r_y"]], dtype=float)
ball_x = np.array([np.nan if v in ("", None) else float(v) for v in data["ball_x"]], dtype=float)
ball_y = np.array([np.nan if v in ("", None) else float(v) for v in data["ball_y"]], dtype=float)
size_x = np.array([np.nan if v in ("", None) else float(v) for v in data["size_x"]], dtype=float)
size_y = np.array([np.nan if v in ("", None) else float(v) for v in data["size_y"]], dtype=float)
conf = np.array([np.nan if v in ("", None) else float(v) for v in data["conf"]], dtype=float)
proximity_instep = np.array([np.nan if v in ("", None) else float(v) for v in data["proximity_instep"]], dtype=float)
proximity_knee = np.array([np.nan if v in ("", None) else float(v) for v in data["proximity_knee"]], dtype=float)
proximity_wrist = np.array([np.nan if v in ("", None) else float(v) for v in data["proximity_wrist"]], dtype=float)
#$touch = np.array([np.nan if v == "" else bool(int(v)) for v in data["touch"]], dtype=float)

# smooth 
smoothing_window = 3
smoothed_instep_left_x = np.convolve(instep_left_x, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_instep_left_y = np.convolve(instep_left_y, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_instep_right_x = np.convolve(instep_right_x, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_instep_right_y = np.convolve(instep_right_y, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_knee_left_x = np.convolve(knee_left_x, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_knee_left_y = np.convolve(knee_left_y, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_knee_right_x = np.convolve(knee_right_x, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_knee_right_y = np.convolve(knee_right_y, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_ankle_left_x = np.convolve(ankle_left_x, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_ankle_left_y = np.convolve(ankle_left_y, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_ankle_right_x = np.convolve(ankle_right_x, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_ankle_right_y = np.convolve(ankle_right_y, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_wrist_left_x = np.convolve(wrist_left_x, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_wrist_left_y = np.convolve(wrist_left_y, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_wrist_right_x = np.convolve(wrist_right_x, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_wrist_right_y = np.convolve(wrist_right_y, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_ball_x = np.convolve(ball_x, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_ball_y = np.convolve(ball_y, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_size_x = np.convolve(size_x, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_size_y = np.convolve(size_y, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_conf = np.convolve(conf, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_proximity_instep = np.convolve(proximity_instep, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_proximity_knee = np.convolve(proximity_knee, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_proximity_wrist = np.convolve(proximity_wrist, np.ones(smoothing_window)/smoothing_window, mode='valid')

# make it cartesian
cartesian_instep_left_y = np.array([(height - 1) - smoothed_instep_left_y[i] for i in range(len(smoothed_instep_left_y))])
cartesian_instep_right_y = np.array([(height - 1) - smoothed_instep_right_y[i] for i in range(len(smoothed_instep_right_y))])
cartesian_knee_left_y = np.array([(height - 1) - smoothed_knee_left_y[i] for i in range(len(smoothed_knee_left_y))])
cartesian_knee_right_y = np.array([(height - 1) - smoothed_knee_right_y[i] for i in range(len(smoothed_knee_right_y))])
cartesian_ankle_left_y = np.array([(height - 1) - smoothed_ankle_left_y[i] for i in range(len(smoothed_ankle_left_y))])
cartesian_ankle_right_y = np.array([(height - 1) - smoothed_ankle_right_y[i] for i in range(len(smoothed_ankle_right_y))])
cartesian_wrist_left_y = np.array([(height - 1) - smoothed_wrist_left_y[i] for i in range(len(smoothed_wrist_left_y))])
cartesian_wrist_right_y = np.array([(height - 1) - smoothed_wrist_right_y[i] for i in range(len(smoothed_wrist_right_y))])
cartesian_ball_y = np.array([(height - 1) - smoothed_ball_y[i] for i in range(len(smoothed_ball_y))])

#velocity
dt = 1/fps
velocity_instep_left_x = np.diff(smoothed_instep_left_x)/dt
velocity_instep_left_y = np.diff(cartesian_instep_left_y)/dt
velocity_instep_right_x = np.diff(smoothed_instep_right_x)/dt
velocity_instep_right_y = np.diff(cartesian_instep_right_y)/dt
velocity_knee_left_x = np.diff(smoothed_knee_left_x)/dt
velocity_knee_left_y = np.diff(cartesian_knee_left_y)/dt
velocity_knee_right_x = np.diff(smoothed_knee_right_x)/dt
velocity_knee_right_y = np.diff(cartesian_knee_right_y)/dt
velocity_ankle_left_x = np.diff(smoothed_ankle_left_x)/dt
velocity_ankle_left_y = np.diff(cartesian_ankle_left_y)/dt
velocity_ankle_right_x = np.diff(smoothed_ankle_right_x)/dt
velocity_ankle_right_y = np.diff(cartesian_ankle_right_y)/dt
velocity_wrist_left_x = np.diff(smoothed_wrist_left_x)/dt
velocity_wrist_left_y = np.diff(cartesian_wrist_left_y)/dt
velocity_wrist_right_x = np.diff(smoothed_wrist_right_x)/dt
velocity_wrist_right_y = np.diff(cartesian_wrist_right_y)/dt
velocity_ball_x = np.diff(smoothed_ball_x)/dt
velocity_ball_y = np.diff(cartesian_ball_y)/dt

last_juggle_time = 0
juggle_count = 0
apex = False
idle = True
pending = False
inflight = False
Carry = False
juggle_streak = 0
juggles = []
streaks = []

for frame in range(len(t) - 2 ):
    # debug/preconditioning
    if frame == 0: continue
    #if t[frame] < 19.6: continue
    #if t[frame] > 30.4: continue

    # ref acceleration
    ref_acc = -47

    # teleport
    D = smoothed_size_y[frame]
    if np.isnan(D) or conf[frame] < 0.2:
        continue
    k_jump_y = abs(cartesian_ball_y[frame] - cartesian_ball_y[frame - 1]) / D
    k_jump_x = abs(smoothed_ball_x[frame] - smoothed_ball_x[frame - 1]) / D 
    k_jump_d = abs(D - smoothed_size_y[frame - 1]) / D
    k_jump_pos_thresh = 0.65
    k_jump_d_thresh = 0.2 # found .15 but 
    if k_jump_x > k_jump_pos_thresh or k_jump_y > k_jump_pos_thresh or k_jump_d > k_jump_d_thresh:
        velocity_ball_y[frame] = np.nan
        velocity_ball_y[frame - 1] = np.nan
        juggle_streak = 0
        continue

    #cooldown - currently 0.3, but change of number maybe needed.
    if (t[frame] - last_juggle_time) < 0.29:
        continue
    
    # above ground check
    above_thresh = 0.3
    above_window = slice(frame - max(1, round(above_thresh * fps)), frame)
    above = np.nanmin(cartesian_ball_y[above_window]) - np.nanmin(np.array([cartesian_ankle_left_y[above_window], cartesian_ankle_right_y[above_window]]))
    if not above > 0:
        juggle_streak = 0
        continue
    
    # neg change of vel
    if not (velocity_ball_y[frame - 1] <= 0 and velocity_ball_y[frame + 1] > 0):
        continue

    window = slice(frame, frame + max(1, round(0.1 * fps)))
    soon = min(
        np.nanmin(smoothed_proximity_instep[window] / smoothed_size_y[window]),
        np.nanmin(smoothed_proximity_knee[window] / smoothed_size_y[window]),
    )
    if not (soon < 0.75):
        continue
    
    # foot velocity check
    vy_window = slice(frame - max(1, round(0.1 * fps)), frame + max(1, round(0.1 * fps)))
    vly_instep = np.nanmax(velocity_instep_left_y[vy_window] / smoothed_size_y[vy_window])
    vry_instep = np.nanmax(velocity_instep_right_y[vy_window] / smoothed_size_y[vy_window])
    vel_l_check = (vly_instep > 1.8)
    vel_r_check = (vry_instep > 1.8)
    
    if not (vel_l_check or vel_r_check):
        juggle_streak = 0
        continue

    # ball leaving chec
    bl_window1 = max(1, round(0.27 * fps))
    bl_window2 = max(1, round(0.03 * fps))
    vftvsb = slice(frame + bl_window2, frame + bl_window1 + bl_window2)
    ball_higher = []
    for i in range(frame + bl_window2, frame + bl_window1 + bl_window2):
        if vel_l_check and vel_r_check:
            foot_v = max(velocity_instep_left_y[i] / smoothed_size_y[i], velocity_instep_right_y[i] / smoothed_size_y[i])
        elif vel_l_check:
            foot_v = velocity_instep_left_y[i] / smoothed_size_y[i]
        else:
            foot_v = velocity_instep_right_y[i] / smoothed_size_y[i]
        ball_v = velocity_ball_y[i] / smoothed_size_y[i]
        if not np.isfinite(ball_v) or not np.isfinite(foot_v):
            continue
        if ball_v > foot_v:
            ball_higher.append(1)
        else:
            ball_higher.append(0)
    
    if not ball_higher.count(1) / len(ball_higher) > 0.70:
        juggle_streak = 0
        continue
    
    # depth (r)
    shin_left_x = smoothed_ankle_left_x[frame] - smoothed_knee_left_x[frame]
    shin_left_y = cartesian_ankle_left_y[frame] - cartesian_knee_left_y[frame]
    shin_right_x = smoothed_ankle_right_x[frame] - smoothed_knee_right_x[frame]
    shin_right_y = cartesian_ankle_right_y[frame] - cartesian_knee_right_y[frame]
    
    shin_left_length = np.sqrt(shin_left_x**2 + shin_left_y**2)
    shin_right_length = np.sqrt(shin_right_x**2 + shin_right_y**2)
    
    if vel_l_check and vel_r_check:
        if vel_l_check > vel_r_check:
            shin = shin_left_length
        else:
            shin = shin_right_length
    elif vel_l_check:
        shin = shin_left_length
    else:
        shin = shin_right_length
    
    r = D/shin

    if r > 1.2:
        juggle_streak = 0
        continue

    # ball apex
    apex_y = cartesian_ball_y[frame]
    for i in range(frame + 1, len(velocity_ball_y)):
        if not np.isfinite(velocity_ball_y[i]):
            continue
        if velocity_ball_y[i] < 0:
            break
        if cartesian_ball_y[i] > apex_y:
            apex_y = cartesian_ball_y[i]
    climb = (apex_y - cartesian_ball_y[frame]) / D

    if not climb > 0.45:
        continue

    # hand proximity check
    hand_window = slice(frame - max(1, round(0.1 * fps)), frame + max(1, round(0.6 * fps)))
    hand = []
    hand_near = False
    hand_touch = False
    hand_grab = False
    for i in range(frame - max(1, round(0.1 * fps)), frame + max(1, round(0.6 * fps))):
        if not np.isfinite(smoothed_proximity_wrist[i]):
            continue
        if smoothed_proximity_wrist[i] / smoothed_size_y[i] > 0.5:
            continue
        if smoothed_proximity_wrist[i] / smoothed_size_y[i] > smoothed_proximity_knee[i] / smoothed_size_y[i] or smoothed_proximity_wrist[i] / smoothed_size_y[i] > smoothed_proximity_instep[i] / smoothed_size_y[i]:
            continue
        hand_near = True
        if not abs(velocity_wrist_left_y[i] / smoothed_size_y[i]) > 0 or abs(velocity_wrist_right_y[i] / smoothed_size_y[i]) > 0:
            continue
        hand_near = True
        if not ((abs(velocity_wrist_left_y[i] / smoothed_size_y[i]) + abs(velocity_ball_y[i] / smoothed_size_y[i]))/ abs(velocity_ball_y[i] / smoothed_size_y[i]) > 1.2 
            or (abs(velocity_wrist_right_y[i] / smoothed_size_y[i]) + abs(velocity_ball_y[i] / smoothed_size_y[i]))/ abs(velocity_ball_y[i] / smoothed_size_y[i]) > 1.2):
            continue
        hand_touch = True
        break
    if hand_touch or hand_grab:
        juggle_streak = 0
        continue

    # TODO: add knees, head. Also add distance 
    
    juggle_count += 1
    juggle_streak += 1
    last_juggle_time = t[frame]
    print(f"Juggle at time {t[frame]:.2f}, streak is {juggle_streak}")
    juggles.append(t[frame])
    streaks.append((juggle_streak, t[frame]))

print(juggle_count)

stem = os.path.splitext(os.path.basename(video_path))[0]
output_path = f"outputs/touch_pose_{stem}.mp4"
cap = cv2.VideoCapture(video_path)
proximity_file = open(f"outputs/proximity_{stem}.csv", "w")

print("Video opened:", cap.isOpened())

frame_number = 0 
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)
print("Writer opened:", out.isOpened())

with mp_pose.Pose() as pose:
    jugglecooloff = 0
    while True:

        success, frame = cap.read()

        if not success:
            break

        if not already_upright and roll in quarter_turn:
            frame = cv2.rotate(frame, quarter_turn[roll])

        t = frame_number / fps

        results = model(frame, verbose=False)

        ball_found = False

        if t in juggles:
            streak = streaks[juggles.index(t)][0]
            color = (0, 255, 0)
            foot_color = (0, 255, 0)
            jugglecooloff = t
        else:
            if t - jugglecooloff < 0.1:
                color = (0, 255, 0)
                foot_color = (0, 255, 0)
            else:
                color = (0, 105, 0)
                foot_color = (0, 105, 0)



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
                    color,
                    4
                )

                cv2.circle(
                    frame,
                    (int(center_x), int(center_y)),
                    8,
                    color,
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

            wrist_left = pose_results.pose_landmarks.landmark[15]
            wrist_right = pose_results.pose_landmarks.landmark[16]

            wrist_left_x = int(wrist_left.x * frame.shape[1])
            wrist_left_y = int(wrist_left.y * frame.shape[0])

            wrist_right_x = int(wrist_right.x * frame.shape[1])
            wrist_right_y = int(wrist_right.y * frame.shape[0])

            instep_left_x = ankle_left_x + (toe_left_x - ankle_left_x) * 0.6
            instep_left_y = ankle_left_y + (toe_left_y - ankle_left_y) * 0.6
            instep_right_x = ankle_right_x + (toe_right_x - ankle_right_x) * 0.6
            instep_right_y = ankle_right_y + (toe_right_y - ankle_right_y) * 0.6

            cv2.circle(
                frame,
                (toe_left_x, toe_left_y),
                8,
                foot_color,
                -1
            )
            cv2.circle(
                frame,
                (toe_right_x, toe_right_y),
                8,
                foot_color,
                -1
            )

            cv2.circle(
                frame,
                (ankle_left_x, ankle_left_y),
                8,
                foot_color,
                -1
            )

            cv2.circle(
                frame,
                (ankle_right_x, ankle_right_y),
                8,
                foot_color,
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
                (wrist_left_x, wrist_left_y),
                8,
                (10, 100, 255),
                -1
            )

            cv2.circle(
                frame,
                (wrist_right_x, wrist_right_y),
                8,
                (10, 100, 255),
                -1
            )

            cv2.circle(
                frame,
                (int(instep_left_x), int(instep_left_y)),
                8,
                foot_color,
                -1
            )

            cv2.circle( 
                frame,
                (int(instep_right_x), int(instep_right_y)),
                8,
                foot_color,
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
            wrist_left_x = None
            wrist_left_y = None
            wrist_right_x = None
            wrist_right_y = None

        cv2.putText(
            frame,
            f"Time: {t:.2f}s",
            (100, 1600),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        if jugglecooloff != 0:
            cv2.putText(
                frame,
                f"Juggle {juggles.index(jugglecooloff) + 1}, Streak {streak}",
                (300, 1400),
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
            proximity_wrist_left_x = abs(wrist_left_x - center_x)
            proximity_wrist_left_y = abs(wrist_left_y - center_y)
            proximity_wrist_right_x = abs(wrist_right_x - center_x)
            proximity_wrist_right_y = abs(wrist_right_y - center_y)
        else:
            proximity = False
            touch = False

        
        
        out.write(frame)

        frame_number += 1

        if frame_number % max(1, int(fps)) == 0:
            print("Processed frames:", frame_number)

out.release()


print("Finished!")
print("Frames processed:", frame_number)
print("Output:", output_path)