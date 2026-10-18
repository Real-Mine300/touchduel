import os
import cv2
import mediapipe as mp
from ultralytics import YOLO
import csv
import numpy as np
import json
import sys


os.makedirs("outputs", exist_ok=True)

video_path = sys.argv[1] if len(sys.argv) > 1 else "recordings/Rain_3.mov"
stem = os.path.splitext(os.path.basename(video_path))[0]
output_path = f"outputs/kalman_overlay_{stem}.mp4"
cap = cv2.VideoCapture(video_path)
proximity_file = open(f"outputs/proximity_{stem}.csv", "w")

print("Video opened:", cap.isOpened())

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

def report(done):
    half = total_frames if total_frames > 0 else max(done, 1)
    print(f"PROGRESS {int(done)} {int(half) * 2}", flush=True)

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

model = YOLO("yolov8n.pt")

BALL_CLASS = 32

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

frame_number = 0

n_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

pose_lm = np.full((n_frames_est, 33, 4), np.nan, dtype=np.float32)

writer = csv.writer(proximity_file)
writer.writerow(["t", "instep_l_x", "instep_l_y", "instep_r_x", "instep_r_y",
 "knee_l_x", "knee_l_y", "knee_r_x", "knee_r_y", "ankle_l_x", "ankle_l_y", "ankle_r_x", "ankle_r_y",
 "wrist_l_x", "wrist_l_y", "wrist_r_x", "wrist_r_y", "shoulders_l_x", "shoulders_l_y", "shoulders_r_x", "shoulders_r_y", "hips_l_x", "hips_l_y", "hips_r_x", "hips_r_y",
  "ball_x", "ball_y", "size_x", "size_y", "conf", "touch"])

data = {
    "t": [],
    "instep_l_x": [], "instep_l_y": [], "instep_r_x": [], "instep_r_y": [],
    "knee_l_x": [], "knee_l_y": [], "knee_r_x": [], "knee_r_y": [],
    "ankle_l_x": [], "ankle_l_y": [], "ankle_r_x": [], "ankle_r_y": [],
    "wrist_l_x": [], "wrist_l_y": [], "wrist_r_x": [], "wrist_r_y": [],
    "shoulders_l_x": [], "shoulders_l_y": [], "shoulders_r_x": [], "shoulders_r_y": [],
    "hips_l_x": [], "hips_l_y": [], "hips_r_x": [], "hips_r_y": [],
    "ball_x": [], "ball_y": [], "size_x": [], "size_y": [], "conf": [], "touch": []
}

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

    memory_ball_h = None
    memory_ball_x = None
    memory_ball_y = None
    memory_shoulders_left_x = None
    memory_body_width = None
    memory_center_body_x = None
    memory_center_body_y = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        if not already_upright and roll in quarter_turn:
            frame = cv2.rotate(frame, quarter_turn[roll])
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

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

        cam_dx_list.append(cam_dx)
        cam_dy_list.append(cam_dy)

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
                if memory_ball_y is None:
                    left, top, right, bottom = (
                        boxes.xyxy[best_index].tolist()
                    )
                    center_x = (left + right) / 2
                    center_y = (top + bottom) / 2
                    memory_ball_h = bottom - top
                    memory_ball_y = center_y
                    memory_ball_x = center_x
                    ball_found = True
                else:
                    passed_boxes = []
                    for bal in ball_indices:
                        left, top, right, bottom = boxes.xyxy[bal].tolist()
                        center_x = (left + right) / 2
                        center_y = (top + bottom) / 2
                        height_check = memory_ball_h / (bottom - top)
                        distance_traveled = ((abs(memory_ball_x - center_x) ** 2 + abs(memory_ball_y - center_y) ** 2) ** 0.5) / memory_ball_h
                        #if distance_traveled <= 2 and .5 <= height_check <= 2:
                        if True:
                            ball_found = True
                            passed_boxes.append(bal)
                    if len(passed_boxes) < 1:
                        ball_found = False
                        center_x = None
                        center_y = None
                        boxes = None
                        best_index = None
                        left = None
                        top = None
                        right = None
                        bottom = None
                    else:
                        lowest_diff = None
                        lowest_diff_ind = None
                        for bal in passed_boxes:
                            left, top, right, bottom = boxes.xyxy[bal].tolist()
                            center_x = (left + right) / 2
                            center_y = (top + bottom) / 2
                            distance_traveled = ((abs(memory_ball_x - center_x) ** 2 + abs(memory_ball_y - center_y) ** 2) ** 0.5) / memory_ball_h
                            if lowest_diff is None or distance_traveled < lowest_diff:
                                lowest_diff = distance_traveled
                                lowest_diff_ind = bal
                        best_index = lowest_diff_ind
                        left, top, right, bottom = boxes.xyxy[best_index].tolist()
                        center_x = (left + right) / 2
                        center_y = (top + bottom) / 2
                        memory_ball_h = bottom - top
                        memory_ball_y = center_y
                        memory_ball_x = center_x
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

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_results = pose.process(rgb_frame)

        if pose_results.pose_landmarks:

            if frame_number < n_frames_est:
                for j, lm in enumerate(pose_results.pose_landmarks.landmark):
                    pose_lm[frame_number, j] = (lm.x, lm.y, lm.z, lm.visibility)
            
            shoulders_left = pose_results.pose_landmarks.landmark[11]
            shoulders_right = pose_results.pose_landmarks.landmark[12]
            shoulders_left_x = int(shoulders_left.x * frame.shape[1])
            shoulders_left_y = int(shoulders_left.y * frame.shape[0])
            shoulders_right_x = int(shoulders_right.x * frame.shape[1])
            shoulders_right_y = int(shoulders_right.y * frame.shape[0])

            hips_left = pose_results.pose_landmarks.landmark[23]
            hips_right = pose_results.pose_landmarks.landmark[24]
            hips_left_x = int(hips_left.x * frame.shape[1])
            hips_left_y = int(hips_left.y * frame.shape[0])
            hips_right_x = int(hips_right.x * frame.shape[1])
            hips_right_y = int(hips_right.y * frame.shape[0])
            
            jump_between_x = None
            jump_between_y = None
            if (shoulders_right_x and shoulders_left_x):
                center_body_x = abs(shoulders_left_x + shoulders_right_x) / 2
                center_body_y = abs(shoulders_left_y + shoulders_right_y) / 2
                if not memory_center_body_x:
                    memory_center_body_x = center_body_x
                    memory_center_body_y = center_body_y
                else:
                    jump_between_x = abs(center_body_x - memory_center_body_x)
                    jump_between_y = abs(center_body_y - memory_center_body_y)
            else:
                center_body_x = None
                center_body_y = None
            jump_thresh = 6.7
            if memory_ball_h and jump_between_x:
                pose_check = (jump_between_x ** 2 + jump_between_y ** 2) ** 0.5 / memory_ball_h > jump_thresh
            elif not center_body_x:
                pose_check = False
            else:
                pose_check = False
            pose_check = False
            if not pose_check:
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

                memory_center_body_x = center_body_x
                memory_center_body_y = center_body_y
            else:
                ankle_left_x = None
                ankle_left_y = None
                ankle_right_x = None
                ankle_right_y = None
                toe_left_x = None
                toe_left_y = None
                toe_right_x = None
                toe_right_y = None
                knee_left_x = None
                knee_left_y = None
                knee_right_x = None
                knee_right_y = None
                wrist_left_x = None
                wrist_left_y = None
                wrist_right_x = None
                wrist_right_y = None
                instep_left_x = None
                instep_left_y = None
                instep_right_x = None
                instep_right_y = None
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
            shoulders_left_x = None
            shoulders_left_y = None
            shoulders_right_x = None
            shoulders_right_y = None
            hips_left_x = None
            hips_left_y = None
            hips_right_x = None
            hips_right_y = None
            center_body_x = None
            center_body_y = None
        if ball_found and pose_results.pose_landmarks:
            if (instep_left_x and center_body_x):
                proximity_right_x = abs(instep_right_x - center_x)
                proximity_right_y = abs(instep_right_y - center_y)
                proximity_left_x = abs(instep_left_x - center_x)
                proximity_left_y = abs(instep_left_y - center_y)
                proximity_shoulders_left_x = abs(shoulders_left_x - center_x)
                proximity_shoulders_left_y = abs(shoulders_left_y - center_y)
                proximity_shoulders_right_x = abs(shoulders_right_x - center_x)
                proximity_shoulders_right_y = abs(shoulders_right_y - center_y)
                proximity_hips_left_x = abs(hips_left_x - center_x)
                proximity_hips_left_y = abs(hips_left_y - center_y)
                proximity_hips_right_x = abs(hips_right_x - center_x)
                proximity_hips_right_y = abs(hips_right_y - center_y)
                proximity_knee_right_x = abs(knee_right_x - center_x)
                proximity_knee_right_y = abs(knee_right_y - center_y)
                proximity_knee_left_x = abs(knee_left_x - center_x)
                proximity_knee_left_y = abs(knee_left_y - center_y)
                proximity_wrist_left_x = abs(wrist_left_x - center_x)
                proximity_wrist_left_y = abs(wrist_left_y - center_y)
                proximity_wrist_right_x = abs(wrist_right_x - center_x)
                proximity_wrist_right_y = abs(wrist_right_y - center_y)
                size_y = bottom - top
                size_x = right - left
                touch_right = proximity_right_x <= size_x and proximity_right_y <= size_y
                touch_left = proximity_left_x <= size_x and proximity_left_y <= size_y
                touch = touch_right or touch_left
            else:
                touch = False
        else:
            touch = False

        if ball_found and pose_results.pose_landmarks:
            if center_body_x and instep_left_x:
                row =[
                    t, instep_left_x, instep_left_y, instep_right_x, instep_right_y,
                    knee_left_x, knee_left_y, knee_right_x, knee_right_y,
                    ankle_left_x, ankle_left_y, ankle_right_x, ankle_right_y,
                    wrist_left_x, wrist_left_y, wrist_right_x, wrist_right_y,
                    shoulders_left_x, shoulders_left_y, shoulders_right_x, shoulders_right_y,
                    hips_left_x, hips_left_y, hips_right_x, hips_right_y,
                    center_x, center_y, size_x, size_y, float(boxes.conf[best_index]), touch,
                    ]
        elif not ball_found and pose_results.pose_landmarks:
            row = [t, instep_left_x, instep_left_y, instep_right_x,
            instep_right_y, knee_left_x, knee_left_y, knee_right_x, knee_right_y,
            ankle_left_x, ankle_left_y, ankle_right_x, ankle_right_y, 
            wrist_left_x, wrist_left_y, wrist_right_x, wrist_right_y,
            shoulders_left_x, shoulders_left_y, shoulders_right_x, shoulders_right_y,
            hips_left_x, hips_left_y, hips_right_x, hips_right_y,
            None, None, None, None, None, None, None, None, None]
        elif ball_found and not pose_results.pose_landmarks:
            row = [t, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, center_x, center_y, size_x, size_y, float(boxes.conf[best_index]), None]
        else:
            row = [t, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
        
        
        writer.writerow(row)

        for key, val in zip(data.keys(), row):
            data[key].append(val)

        frame_number += 1

        if frame_number % 10 == 0:
            report(frame_number)

        if frame_number % max(1, int(fps)) == 0:
            print("Processed frames:", frame_number)

cap.release()
pose_lm = pose_lm[:frame_number]
np.savez_compressed(f"outputs/cache_{stem}.npz", pose_lm=pose_lm, fps=fps, width=width, height=height)

print("Finished!")
print("Frames processed:", frame_number)

if total_frames <= 0:
    total_frames = frame_number
report(frame_number)

print("Output:", output_path)
print("Proximity file:", proximity_file.name)
proximity_file.close()

#do I need this?
#with open(proximity_file.name, mode='r', newline='', encoding='utf-8') as file:
#    reader = csv.DictReader(file)
#    
#    data = {header: [] for header in reader.fieldnames}
#    for row in reader:
#        for header in reader.fieldnames:
#            data[header].append(row[header])

cam_dx = np.array(cam_dx_list, dtype=float)
cam_dy = np.array(cam_dy_list, dtype=float)
step_x = np.nan_to_num(cam_dx, nan=0.0)
step_y = np.nan_to_num(cam_dy, nan=0.0)
cam_pos_x = np.cumsum(step_x)
cam_pos_y = np.cumsum(step_y)

t = np.array([float(v) for v in data["t"]], dtype=float)

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
shoulders_left_x = np.array([np.nan if v in ("", None) else float(v) for v in data["shoulders_l_x"]], dtype=float)
shoulders_left_y = np.array([np.nan if v in ("", None) else float(v) for v in data["shoulders_l_y"]], dtype=float)
shoulders_right_x = np.array([np.nan if v in ("", None) else float(v) for v in data["shoulders_r_x"]], dtype=float)
shoulders_right_y = np.array([np.nan if v in ("", None) else float(v) for v in data["shoulders_r_y"]], dtype=float)
hips_left_x = np.array([np.nan if v in ("", None) else float(v) for v in data["hips_l_x"]], dtype=float)
hips_left_y = np.array([np.nan if v in ("", None) else float(v) for v in data["hips_l_y"]], dtype=float)
hips_right_x = np.array([np.nan if v in ("", None) else float(v) for v in data["hips_r_x"]], dtype=float)
hips_right_y = np.array([np.nan if v in ("", None) else float(v) for v in data["hips_r_y"]], dtype=float)
ball_x = np.array([np.nan if v in ("", None) else float(v) for v in data["ball_x"]], dtype=float)
ball_y = np.array([np.nan if v in ("", None) else float(v) for v in data["ball_y"]], dtype=float)
size_x = np.array([np.nan if v in ("", None) else float(v) for v in data["size_x"]], dtype=float)
size_y = np.array([np.nan if v in ("", None) else float(v) for v in data["size_y"]], dtype=float)
conf = np.array([np.nan if v in ("", None) else float(v) for v in data["conf"]], dtype=float)
#$touch = np.array([np.nan if v == "" else bool(int(v)) for v in data["touch"]], dtype=float)

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
shoulders_left_x -= cam_pos_x
shoulders_left_y -= cam_pos_y
shoulders_right_x -= cam_pos_x
shoulders_right_y -= cam_pos_y
hips_left_x -= cam_pos_x
hips_left_y -= cam_pos_y
hips_right_x -= cam_pos_x
hips_right_y -= cam_pos_y

# filling in + predictions
filled_in_frames = max(1, round(fps * 0.1))
gap_start = None
gaps = []
for i in range(0, len(ball_x)):
    if not np.isfinite(ball_x[i]) and gap_start is None:
        gap_start = i
    if np.isfinite(ball_x[i]) and gap_start is not None:
        gaps.append([gap_start, i, i - gap_start])
        gap_start = None
if gap_start is not None:
    gaps.append([gap_start, len(ball_x), len(ball_x) - gap_start])
print(gaps)
#print(size_y[141:148])

kalman_frames = set()
for q,r,l in gaps:
    if q > 0 and l <= filled_in_frames and r < len(ball_x):
        kalman_frames.update(range(q,r))
        for k in range(q,r):
            size_y[k] = np.interp(k, [q-1, r], [size_y[q-1], size_y[r]])
            size_x[k] = np.interp(k, [q-1, r], [size_x[q-1], size_x[r]])
            conf[k] = np.interp(k, [q-1, r], [conf[q-1], conf[r]])
#print(sorted(kalman_frames))
#print(size_y[141:148])
kf = cv2.KalmanFilter(4,2)
kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], np.float32)
kf.measurementNoiseCov = np.eye(2,dtype=np.float32) * 10
kf.processNoiseCov = np.diag([1, 1, 10000, 10000]).astype(np.float32)
#print(kf.measurementNoiseCov)
#print(kf.processNoiseCov)
#print(kf.processNoiseCov.dtype)

print(np.isfinite(ball_x).sum())
print(ball_y[121:125])
#teach kalman
active = False
for i in range(len(ball_x)):
    if np.isfinite(ball_x[i]):
        if not active:
            kf.statePost = np.array([[ball_x[i]], [ball_y[i]], [0], [0]], np.float32)
            kf.errorCovPost = np.diag([10, 10, 1000000, 1000000]).astype(np.float32)
            active= True
        else:
            dt = t[i] - t[i - 1]
            forward_guess = np.eye(4, dtype=np.float32)
            forward_guess[0, 2] = dt
            forward_guess[1, 3] = dt
            kf.transitionMatrix = forward_guess
            kf.predict()
            kf.correct(np.array([[ball_x[i]],[ball_y[i]]], np.float32))
    elif i in kalman_frames:
        dt = t[i] - t[i - 1]
        forward_guess = np.eye(4, dtype=np.float32)
        forward_guess[0, 2] = dt
        forward_guess[1, 3] = dt
        kf.transitionMatrix = forward_guess
        pred = kf.predict()
        ball_x[i] = pred[0, 0]
        ball_y[i] = pred[1, 0]
    else:
        active = False
print(np.isfinite(ball_x).sum())

# proximities
parts = {
    'instep' : (instep_left_x, instep_left_y, instep_right_x, instep_right_y),
    'knee' : (knee_left_x, knee_left_y, knee_right_x, knee_right_y),
    'ankle' : (ankle_left_x, ankle_left_y, ankle_right_x, ankle_right_y),
    'wrist' : (wrist_left_x, wrist_left_y, wrist_right_x, wrist_right_y),
    'shoulders' : (shoulders_left_x, shoulders_left_y, shoulders_right_x, shoulders_right_y),
    'hip' : (hips_left_x, hips_left_y, hips_right_x, hips_right_y),
}
proximity = {}

for name, (lx, ly, rx, ry) in parts.items():
    left = np.hypot(lx - ball_x, ly - ball_y)
    right = np.hypot(rx - ball_x, ry - ball_y)
    proximity[name] = np.minimum(left,right)


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
smoothed_proximity_instep = np.convolve(proximity['instep'], np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_proximity_knee = np.convolve(proximity['knee'], np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_proximity_ankle = np.convolve(proximity['ankle'], np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_proximity_wrist = np.convolve(proximity['wrist'], np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_proximity_shoulders = np.convolve(proximity['shoulders'], np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_proximity_hip = np.convolve(proximity['hip'], np.ones(smoothing_window)/smoothing_window, mode='valid')

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
dt = np.diff(t[1:-1])
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
missing = 0
last_corner = -1
last_left_foot_frame = -1
last_left_foot_x = None
last_left_foot_y = None
last_right_foot_frame = -1 
last_right_foot_x = None 
last_right_foot_y = None
last_left_knee_frame = -1
last_left_knee_x = None
last_left_knee_y = None
last_right_knee_frame = -1
last_right_knee_x = None
last_right_knee_y = None
last_left_wrist_frame = -1
last_left_wrist_x = None
last_left_wrist_y = None
last_right_wrist_frame = -1
last_right_wrist_x = None
last_right_wrist_y = None
juggles = []
streaks = []
labels = []
juggle_times = []

for frame in range(len(t) - 2 ):
    # debug/preconditioning
    if frame == 0: continue
    #if t[frame] < 19.6: continue
    #if t[frame] > 30.4: continue


    # missing ball
    D = smoothed_size_y[frame]
    if np.isnan(D) or conf[frame] < 0.2:
        missing += 1
        if missing > round(fps/3):
            juggle_streak = 0
            idle = True
            pending = False
            inflight = False
            Carry = False
        continue
    else:
        missing = 0

    # teleport
    k_jump_y = abs(cartesian_ball_y[frame] - cartesian_ball_y[frame - 1]) / D
    k_jump_x = abs(smoothed_ball_x[frame] - smoothed_ball_x[frame - 1]) / D 
    k_jump_d = abs(D - smoothed_size_y[frame - 1]) / D
    k_jump_pos_thresh = 100 #.65 but 100 for testing
    k_jump_d_thresh = 0.2 # found .15 but 
    if k_jump_x > k_jump_pos_thresh or k_jump_y > k_jump_pos_thresh or k_jump_d > k_jump_d_thresh:
        velocity_ball_y[frame] = np.nan
        velocity_ball_y[frame - 1] = np.nan
        continue

    # update memories
    dist_instep_left = np.hypot(smoothed_instep_left_x[frame] - smoothed_ball_x[frame], cartesian_instep_left_y[frame] - cartesian_ball_y[frame])
    if np.isfinite(dist_instep_left):
        if dist_instep_left > D:
            last_left_foot_x = smoothed_instep_left_x[frame]
            last_left_foot_y = cartesian_instep_left_y[frame]
            last_left_foot_frame = frame
    dist_instep_right = np.hypot(smoothed_instep_right_x[frame] - smoothed_ball_x[frame], cartesian_instep_right_y[frame] - cartesian_ball_y[frame])
    if np.isfinite(dist_instep_right):
        if dist_instep_right > D:
            last_right_foot_x = smoothed_instep_right_x[frame]
            last_right_foot_y = cartesian_instep_right_y[frame]
            last_right_foot_frame = frame
    dist_knee_left = np.hypot(smoothed_knee_left_x[frame] - smoothed_ball_x[frame], cartesian_knee_left_y[frame] - cartesian_ball_y[frame])
    if np.isfinite(dist_knee_left):
        if dist_knee_left > D:
            last_left_knee_x = smoothed_knee_left_x[frame]
            last_left_knee_y = cartesian_knee_left_y[frame]
            last_left_knee_frame = frame
    dist_knee_right = np.hypot(smoothed_knee_right_x[frame] - smoothed_ball_x[frame], cartesian_knee_right_y[frame] - cartesian_ball_y[frame])
    if np.isfinite(dist_knee_right):
        if dist_knee_right > D:
            last_right_knee_x = smoothed_knee_right_x[frame]
            last_right_knee_y = cartesian_knee_right_y[frame]
            last_right_knee_frame = frame
    dist_wrist_left = np.hypot(smoothed_wrist_left_x[frame] - smoothed_ball_x[frame], cartesian_wrist_left_y[frame] - cartesian_ball_y[frame])
    if np.isfinite(dist_wrist_left):
        if dist_wrist_left > D:
            last_left_wrist_x = smoothed_wrist_left_x[frame]
            last_left_wrist_y = cartesian_wrist_left_y[frame]
            last_left_wrist_frame = frame
    dist_wrist_right = np.hypot(smoothed_wrist_right_x[frame] - smoothed_ball_x[frame], cartesian_wrist_right_y[frame] - cartesian_ball_y[frame])
    if np.isfinite(dist_wrist_right):
        if dist_wrist_right > D:
            last_right_wrist_x = smoothed_wrist_right_x[frame]
            last_right_wrist_y = cartesian_wrist_right_y[frame]
            last_right_wrist_frame = frame

    # corners
    if frame < 2 or frame + 1 >= len(velocity_ball_y):
        continue
    if not (np.isfinite(velocity_ball_y[frame]) and np.isfinite(velocity_ball_y[frame - 1]) and np.isfinite(velocity_ball_y[frame + 1]) and np.isfinite(velocity_ball_y[frame - 2])):
        continue
    bend_y = velocity_ball_y[frame] - velocity_ball_y[frame - 1]
    bend_before_y = velocity_ball_y[frame - 1] - velocity_ball_y[frame - 2]
    bend_after_y = velocity_ball_y[frame + 1] - velocity_ball_y[frame]
    corner_y = abs(bend_y) > abs(bend_after_y) + abs(bend_before_y)
    down_to_up = velocity_ball_y[frame - 1] <= 0 and velocity_ball_y[frame + 1] > 0 and velocity_ball_y[frame - 2] < 0
    if down_to_up:
        rise = velocity_ball_y[frame + 1] / D
        if rise < 0.8:
            continue
        if frame - last_corner > round(0.15 * fps):
            recent = round(0.15 * fps)
            label = "unknown"
            best = None
            if last_left_foot_frame != -1 and frame - last_left_foot_frame < recent:
                sigmadistance = np.hypot(last_left_foot_x - smoothed_ball_x[frame], last_left_foot_y - cartesian_ball_y[frame])
                if best is None or sigmadistance < best:
                    best = sigmadistance
                    label = "left_foot"
            if last_right_foot_frame != -1 and frame - last_right_foot_frame < recent:
                sigmadistance = np.hypot(last_right_foot_x - smoothed_ball_x[frame], last_right_foot_y - cartesian_ball_y[frame])
                if best is None or sigmadistance < best:
                    best = sigmadistance
                    label = "right_foot"
            if last_left_knee_frame != -1 and frame - last_left_knee_frame < recent:
                sigmadistance = np.hypot(last_left_knee_x - smoothed_ball_x[frame], last_left_knee_y - cartesian_ball_y[frame])
                if best is None or sigmadistance < best:
                    best = sigmadistance
                    label = "left_knee"
            if last_right_knee_frame != -1 and frame - last_right_knee_frame < recent:
                sigmadistance = np.hypot(last_right_knee_x - smoothed_ball_x[frame], last_right_knee_y - cartesian_ball_y[frame])
                if best is None or sigmadistance < best:
                    best = sigmadistance
                    label = "right_knee"
            if last_left_wrist_frame != -1 and frame - last_left_wrist_frame < recent:
                sigmadistance = np.hypot(last_left_wrist_x - smoothed_ball_x[frame], last_left_wrist_y - cartesian_ball_y[frame])
                if best is not None and sigmadistance < best:
                    best = sigmadistance
                    label = "left_wrist"
            if last_right_wrist_frame != -1 and frame - last_right_wrist_frame < recent:
                sigmadistance = np.hypot(last_right_wrist_x - smoothed_ball_x[frame], last_right_wrist_y - cartesian_ball_y[frame])
                if best is not None and sigmadistance < best:
                    best = sigmadistance
                    label = "right_wrist"
            if label in ["left_foot", "right_foot", "left_knee", "right_knee"]:
                juggle_streak += 1
                last_juggle_time = t[frame]
                juggle_count += 1 # temp count for ease
                juggles.append(frame + 1)          # frame index (the overlay pass needs this)
                juggle_times.append(t[frame])      # seconds (for the JUGGLES output)
                streaks.append(juggle_streak)
                labels.append(label)
            elif label != "unknown":
                juggle_streak = 0
            print(f" Time: {t[frame]:.2f}  corner y: {corner_y} label: {label}, juggle streak: {juggle_streak}")
            last_corner = frame
        continue

print(juggle_count)

stem = os.path.splitext(os.path.basename(video_path))[0]
output_path = f"outputs/kalman_overlay_{stem}.mp4"
cap = cv2.VideoCapture(video_path)
proximity_file = open(f"outputs/proximity_{stem}.csv", "r")

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

def draw_pose(frame, lm, color=(0, 255, 0), min_vis=0.5):
    """lm: (33, 4) array of normalized x, y, z, visibility for one frame."""
    if np.isnan(lm[:, 0]).all():
        return
    h, w = frame.shape[:2]
    pts = {}
    for j in range(33):
        x, y, _, vis = lm[j]
        if np.isfinite(x) and vis >= min_vis:
            pts[j] = (int(x * w), int(y * h))

    for a, b in mp_pose.POSE_CONNECTIONS:      # same bone list draw_landmarks uses
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], color, 3)
    for p in pts.values():
        cv2.circle(frame, p, 5, (0, 0, 255), -1)

cap = cv2.VideoCapture(video_path)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
print("Writer opened:", out.isOpened())

juggle_set = set(juggles)
streak_of = dict(zip(juggles, streaks))
cooloff = max(1, round(fps * 0.1))
last_juggle_frame = None
BONE = (100, 100, 200)

frame_number = 0
while True:
    success, frame = cap.read()
    if not success or frame_number >= len(pose_lm):
        break
    if not already_upright and roll in quarter_turn:
        frame = cv2.rotate(frame, quarter_turn[roll])

    i = frame_number
    h, w = frame.shape[:2]

    # colors
    if i in juggle_set:
        last_juggle_frame = i
    hot = last_juggle_frame is not None and i - last_juggle_frame < cooloff
    if i in kalman_frames:
        color = (255, 205, 0) if hot else (255, 100, 0)
    else:
        color = (0, 255, 0) if hot else (0, 105, 0)

    # skeleton + highlighted joints, all from the cache
    lm = pose_lm[i]
    draw_pose(frame, lm, color=BONE)

    def jp(j):
        if np.isfinite(lm[j, 0]) and lm[j, 3] >= 0.5:
            return (int(lm[j, 0] * w), int(lm[j, 1] * h))
        return None

    for j, c in ((31, color), (32, color), (27, color), (28, color),
                 (25, BONE), (26, BONE), (15, (10, 100, 255)), (16, (10, 100, 255))):
        p = jp(j)
        if p:
            cv2.circle(frame, p, 8, c, -1)

    # instep (60% from ankle to toe)
    for ank, toe in ((27, 31), (28, 32)):
        a, b = jp(ank), jp(toe)
        if a and b:
            cv2.circle(frame, (int(a[0] + (b[0] - a[0]) * 0.6),
                               int(a[1] + (b[1] - a[1]) * 0.6)), 8, color, -1)

    # ball (detections + Kalman fills), add camera offset back
    if i < len(ball_x) and np.isfinite(ball_x[i]) and np.isfinite(ball_y[i]):
        cx = ball_x[i] + cam_pos_x[i]
        cy = ball_y[i] + cam_pos_y[i]
        bw, bh = size_x[i], size_y[i]
        if np.isfinite(bw) and np.isfinite(bh):
            cv2.rectangle(frame, (int(cx - bw / 2), int(cy - bh / 2)),
                          (int(cx + bw / 2), int(cy + bh / 2)), color, 4)
        cv2.circle(frame, (int(cx), int(cy)), 8, color, -1)

    # text
    cv2.putText(frame, f"Time: {i / fps:.2f}s", (50, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if last_juggle_frame is not None:
        n = sum(1 for f in juggles if f <= i)
        cv2.putText(frame, f"Juggle {n}, Streak {streak_of[last_juggle_frame]}",
                    (50, h - 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    out.write(frame)
    frame_number += 1

    if frame_number % 10 == 0:
        report(total_frames + frame_number)

    if frame_number % max(1, int(fps)) == 0:
        print("Processed frames:", frame_number)

out.release()
cap.release()

report(total_frames + frame_number)

rows = []
for i, (tm, lab, st) in enumerate(zip(juggle_times, labels, streaks), start=1):
    rows.append({
        "n": i,
        "time": round(float(tm), 2),
        "label": lab,
        "streak": int(st),
    })
print("JUGGLES " + json.dumps(rows), flush=True)
print("VIDEO " + os.path.abspath(output_path), flush=True)
print("Finished!")
print("Frames processed:", frame_number)
print("Output:", output_path)