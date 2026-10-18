from math import isfinite
import os
import cv2
import mediapipe as mp
from ultralytics import YOLO
import csv
import numpy as np
from collections import Counter


video_path = "recordings/Home_1.mov"
stem = os.path.splitext(os.path.basename(video_path))[0]
output_path = f"outputs/touch_pose_{stem}.mp4"
cap = cv2.VideoCapture(video_path)
proximity_file = open(f"outputs/proximity_{stem}.csv", "r")

print("Video opened:", cap.isOpened())

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("Original res:", width, "x", height)


prev_gray = None
prev_pts = None
cam_dx_list = []
cam_dy_list = []
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)


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
touch = np.array(data['touch'])

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

#clean removed for now

# filling in + predictions
filled_in_frames = max(1, round(fps * 0.25))
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
#print(gaps)
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

#print(np.isfinite(ball_x).sum())
#print(ball_y[121:125])
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
#print(np.isfinite(ball_x).sum())

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

#shins
shin_length_left = np.hypot(
    smoothed_instep_left_x - smoothed_knee_left_x,
    cartesian_instep_left_y - cartesian_knee_left_y,
)
shin_length_right = np.hypot(
    smoothed_instep_right_x - smoothed_knee_right_x,
    cartesian_instep_right_y - cartesian_knee_right_y,
)
hand = np.array([11.81,12.34,13.11,13.71,19.32,19.81,20.24,20.78,27.96,28.38,28.76,
                 36.54,36.96,37.51,38.23,38.70,48.36,48.80,61.35,61.79,62.22,62.72,63.15,
                 68.90,69.59,70.05,79.40,79.84,80.24,87.62,93.81,94.28,94.75,95.65,100.05])
GATES = ["conf/NaN D", "teleport", "4 finite vel", "down_to_up", "rise<0.8",
         "too soon", "label/dist", "shin", "foot_moving", "COUNTED"]
reached = {}
def mark(frame, g):
    for h in hand:
        if abs(t[frame] + 0.1 - h) < 0.2:   # program runs ~0.1s early
            reached[h] = max(reached.get(h, -1), g)
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

for frame in range(len(t) - 2 ):
    # debug/preconditioning
    if frame == 0: continue
    #if t[frame] < 19.6: continue
    #if t[frame] > 30.4: continue
    mark(frame, 0)


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
    mark(frame, 1)
    k_jump_y = abs(cartesian_ball_y[frame] - cartesian_ball_y[frame - 1]) / D
    k_jump_x = abs(smoothed_ball_x[frame] - smoothed_ball_x[frame - 1]) / D 
    k_jump_d = abs(D - smoothed_size_y[frame - 1]) / D
    k_jump_pos_thresh = 100 #.65 but 100 for testing
    k_jump_d_thresh = 100 # found .15 but 
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
    mark(frame, 2)
    if frame < 2 or frame + 1 >= len(velocity_ball_y):
        continue
    mark(frame, 3)
    for h in [19.32, 27.96, 28.38, 36.54, 79.40]:
        if abs(t[frame] + 0.1 - h) < 0.1:
            print(f"{h:.2f} t={t[frame]:.2f} vy={velocity_ball_y[frame-2:frame+2]}")
    if not (np.isfinite(velocity_ball_y[frame]) and np.isfinite(velocity_ball_y[frame - 1]) and np.isfinite(velocity_ball_y[frame + 1]) and np.isfinite(velocity_ball_y[frame - 2])):
        continue
    bounce = False
    juggle = False
    bend_y = velocity_ball_y[frame] - velocity_ball_y[frame - 1]
    bend_before_y = velocity_ball_y[frame - 1] - velocity_ball_y[frame - 2]
    bend_after_y = velocity_ball_y[frame + 1] - velocity_ball_y[frame]
    corner_y = abs(bend_y) > abs(bend_after_y) + abs(bend_before_y)
    down_to_up = velocity_ball_y[frame - 1] <= 0 and velocity_ball_y[frame + 1] > 0 and velocity_ball_y[frame - 2] < 0
    if down_to_up:
        rise = velocity_ball_y[frame + 1] / D
        mark(frame, 4)
        if rise < 0.8:
            continue
        mark(frame, 5)
        if frame - last_corner > round(0.15 * fps):
            mark(frame, 6)
            recent = round(0.15 * fps)
            label = "unknown"
            best = None
            if last_left_foot_frame != -1 and frame - last_left_foot_frame < recent:
                sigmadistance = np.hypot(last_left_foot_x - smoothed_ball_x[frame], last_left_foot_y - cartesian_ball_y[frame]) * 0.90
                if best is None or sigmadistance < best:
                    best = sigmadistance
                    label = "left_foot"
            if last_right_foot_frame != -1 and frame - last_right_foot_frame < recent:
                sigmadistance = np.hypot(last_right_foot_x - smoothed_ball_x[frame], last_right_foot_y - cartesian_ball_y[frame]) * 0.90
                if best is None or sigmadistance < best:
                    best = sigmadistance
                    label = "right_foot"
            if label == "unknown":
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
                    if best is None or sigmadistance < best:
                        best = sigmadistance
                        label = "left_wrist"
                if last_right_wrist_frame != -1 and frame - last_right_wrist_frame < recent:
                    sigmadistance = np.hypot(last_right_wrist_x - smoothed_ball_x[frame], last_right_wrist_y - cartesian_ball_y[frame])
                    if best is None or sigmadistance < best:
                        best = sigmadistance
                        label = "right_wrist"
            
            #dist check
            if label in ["left_foot", "right_foot", "left_knee", "right_knee"] and best <= 1.0 * np.fmax(shin_length_left[frame], shin_length_right[frame]):
                # bounces
                # narrowning bounces
                mark(frame, 7)

                foot_moving = False
                if min(shin_length_left[frame]/D, shin_length_right[frame]/D) < .7:
                    juggle_streak = 0 
                    print(f"time: {t[frame]:.2f}, shin: {min(shin_length_left[frame]/D, shin_length_right[frame]/D) < .7}")
                else:
                    mark(frame, 8)
                    look_ahead = int(round(0.2 * fps))
                    alpha = frame - 1
                    beta = min(alpha + look_ahead, len(velocity_ball_y))
                    half_window = max(1, int(round(0.05 * fps))) #normally .05 but to +.1 i set it at 
                    alpha = max(0, frame - half_window)
                    beta = min(frame + half_window + 1, len(velocity_ball_y))
                    foot_moving = False

                    if label == "left_foot":
                        fx = velocity_instep_left_x[alpha:beta]
                        fy = velocity_instep_left_y[alpha:beta]
                    elif label == "right_foot":
                        fx = velocity_instep_right_x[alpha:beta]
                        fy = velocity_instep_right_y[alpha:beta]
                    else:
                        fx, fy = None, None
                        print(f"time: {t[frame]:.2f}, label: {label}")
                    if fx is not None:
                        bx = velocity_ball_x[alpha:beta]
                        by = velocity_ball_y[alpha:beta]
                        ok = np.all([np.isfinite(fx), np.isfinite(fy), np.isfinite(bx), np.isfinite(by), (by > 0)], axis=0)
                        if ok.any():
                            dot = fx * bx + fy * by
                            ratio = np.abs(fy / by)
                            good = ok & (dot > 0) & (ratio < 10.0)
                            meeting = (fy < 0) & (by > 0)
                            head_on = dot[0] < 0   # or: fy[0] * by[0] < 0 for y-only

                            foot_moving = (meeting.any() or good.any()) and head_on

                            if not foot_moving:
                                print(f"Time: {t[frame]:.2f}, dot: {dot}, meeting: {meeting}, head_on: {head_on}")
                        else:
                            print(f"Time: {t[frame]:.2f}, ok: {ok}")

                    if foot_moving:
                        juggle = False
                        point = 1 # 0 later 1 for testing
                        # point 1 horizontal jump

                        # point 

                        if point > 0:
                            juggle = True
                    else:
                        juggle_streak = 0
                    if juggle:
                        mark(frame, 9)
                        juggle_streak += 1
                        last_juggle_time = t[frame]
                        juggle_count += 1 # temp count for ease
                        juggles.append(frame + 1)
                        streaks.append(juggle_streak)
                        print(f" Time: {t[frame]:.2f}  corner y: {corner_y} label: {label}, juggle streak: {juggle_streak}")
                    
            elif label in ["left_wrist", "right_wrist"]:
                print(f"time: {t[frame]:.2f}, label: {label}")
                juggle_streak = 0
            else:
                print(f"time: {t[frame]:.2f}, label: {label}, best check: {best <= 1.0 * np.fmax(shin_length_left[frame], shin_length_right[frame])}, best: {best}, shin: {np.fmax(shin_length_left[frame], shin_length_right[frame]):.2f}")
            last_corner = frame

print(juggle_count)


for h in hand:
    g = reached.get(h, -1)
    print(f"{h:.2f} died at: {GATES[g] if g >= 0 else 'never reached'}")
print(Counter(GATES[reached.get(h, -1)] if reached.get(h, -1) >= 0 else 'never' for h in hand))