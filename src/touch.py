import os
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import csv

rotateIMG = True

video_path = "recordings/Rain_3.mov"
stem = os.path.splitext(os.path.basename(video_path))[0]
output_path = f"outputs/ball_pose_{stem}.mp4"
cap = cv2.VideoCapture(video_path)
proximity_file = open(f"outputs/proximity_{stem}.csv", "r")

print("Video opened:", cap.isOpened())

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("Original res:", width, "x", height)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")



model = YOLO("yolov8n.pt")

BALL_CLASS = 32

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

frame_number = 0

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
ball_x = np.array(data['ball_x'])
ball_y = np.array(data['ball_y'])
size_x = np.array(data['size_x'])
size_y = np.array(data['size_y'])
conf = np.array(data['conf'])
proximity_instep = np.array(data['proximity_instep'])
proximity_knee = np.array(data['proximity_knee'])
touch = np.array(data['touch'])

t = np.array([float(v) for v in data["t"]], dtype=float)

# acccount for "" into nan
instep_left_x = np.array([np.nan if v == "" else float(v) for v in data["instep_l_x"]], dtype=float)
instep_left_y = np.array([np.nan if v == "" else float(v) for v in data["instep_l_y"]], dtype=float)
instep_right_x = np.array([np.nan if v == "" else float(v) for v in data["instep_r_x"]], dtype=float)
instep_right_y = np.array([np.nan if v == "" else float(v) for v in data["instep_r_y"]], dtype=float)
knee_left_x = np.array([np.nan if v == "" else float(v) for v in data["knee_l_x"]], dtype=float)
knee_left_y = np.array([np.nan if v == "" else float(v) for v in data["knee_l_y"]], dtype=float)
knee_right_x = np.array([np.nan if v == "" else float(v) for v in data["knee_r_x"]], dtype=float)
knee_right_y = np.array([np.nan if v == "" else float(v) for v in data["knee_r_y"]], dtype=float)
ankle_left_x = np.array([np.nan if v == "" else float(v) for v in data["ankle_l_x"]], dtype=float)
ankle_left_y = np.array([np.nan if v == "" else float(v) for v in data["ankle_l_y"]], dtype=float)
ankle_right_x = np.array([np.nan if v == "" else float(v) for v in data["ankle_r_x"]], dtype=float)
ankle_right_y = np.array([np.nan if v == "" else float(v) for v in data["ankle_r_y"]], dtype=float)
ball_x = np.array([np.nan if v == "" else float(v) for v in data["ball_x"]], dtype=float)
ball_y = np.array([np.nan if v == "" else float(v) for v in data["ball_y"]], dtype=float)
size_x = np.array([np.nan if v == "" else float(v) for v in data["size_x"]], dtype=float)
size_y = np.array([np.nan if v == "" else float(v) for v in data["size_y"]], dtype=float)
conf = np.array([np.nan if v == "" else float(v) for v in data["conf"]], dtype=float)
proximity_instep = np.array([np.nan if v == "" else float(v) for v in data["proximity_instep"]], dtype=float)
proximity_knee = np.array([np.nan if v == "" else float(v) for v in data["proximity_knee"]], dtype=float)
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
smoothed_ball_x = np.convolve(ball_x, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_ball_y = np.convolve(ball_y, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_size_x = np.convolve(size_x, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_size_y = np.convolve(size_y, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_conf = np.convolve(conf, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_proximity_instep = np.convolve(proximity_instep, np.ones(smoothing_window)/smoothing_window, mode='valid')
smoothed_proximity_knee = np.convolve(proximity_knee, np.ones(smoothing_window)/smoothing_window, mode='valid')

# make it cartesian
cartesian_instep_left_y = np.array([(height - 1) - smoothed_instep_left_y[i] for i in range(len(smoothed_instep_left_y))])
cartesian_instep_right_y = np.array([(height - 1) - smoothed_instep_right_y[i] for i in range(len(smoothed_instep_right_y))])
cartesian_knee_left_y = np.array([(height - 1) - smoothed_knee_left_y[i] for i in range(len(smoothed_knee_left_y))])
cartesian_knee_right_y = np.array([(height - 1) - smoothed_knee_right_y[i] for i in range(len(smoothed_knee_right_y))])
cartesian_ankle_left_y = np.array([(height - 1) - smoothed_ankle_left_y[i] for i in range(len(smoothed_ankle_left_y))])
cartesian_ankle_right_y = np.array([(height - 1) - smoothed_ankle_right_y[i] for i in range(len(smoothed_ankle_right_y))])
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
velocity_ball_x = np.diff(smoothed_ball_x)/dt
velocity_ball_y = np.diff(cartesian_ball_y)/dt

last_juggle_time = 0
juggle_count = 0
apex = False
idle = True
pending = False
inflight = False
Carry = False
for frame in range(len(t) - 2 ):
    # debug/preconditioning
    if t[frame] < 19.6: continue
    if t[frame] > 30.4: continue

    # teleport
    D = smoothed_size_y[frame]
    if D is np.nan or conf[frame] < 0.2:
        continue
    k_jump_y = abs(cartesian_ball_y[frame] - cartesian_ball_y[frame - 1]) / D
    k_jump_x = abs(smoothed_ball_x[frame] - smoothed_ball_x[frame - 1]) / D 
    k_jump_d = abs(D - smoothed_size_y[frame - 1]) / D
    k_jump_pos_thresh = 0.65
    k_jump_d_thresh = 0.2 # found .15 but 
    if k_jump_x > k_jump_pos_thresh or k_jump_y > k_jump_pos_thresh or k_jump_d > k_jump_d_thresh:
        velocity_ball_y[frame] = np.nan
        velocity_ball_y[frame - 1] = np.nan
        continue

    #cooldown - currently 0.3, but change of number maybe needed.
    if (t[frame] - last_juggle_time) < 0.3:
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
    vel_l_check = (vly_instep > 2)
    vel_r_check = (vry_instep > 2)
    
    if not (vel_l_check or vel_r_check):
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

    # 
    
    juggle_count += 1
    last_juggle_time = t[frame]
    print(f"Juggle at time {t[frame]}, climb is {climb:.4f}")

print(juggle_count)
