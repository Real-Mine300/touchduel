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
idle = False
pending = False
inflight = False
Carry = False
hieghest = 0
hieghesadst = 0
hieghessadst = 0
indeca = 0
for frame in range(len(t) - 2 ):
    #if (t[frame] > 12.7 and t[frame] < 13) or (t[frame] > 20 and t[frame] < 20.3):
        #print("ball: ", velocity_ball_y[frame])
        #print("left: ", velocity_instep_left_y[frame])
        #print("right: ", velocity_instep_right_y[frame])
    if t[frame] < 0.5: continue
    D = smoothed_size_y[frame]
    if D is np.nan or conf[frame] < 0.2:
        continue
    prox_instep = smoothed_proximity_instep[frame]/D
    prox_knee = smoothed_proximity_knee[frame]/D
    if t[frame] <21 and t[frame] :
        print(t[frame])
        print(prox_instep)
        if prox_instep <hieghest:
            hieghest = prox_instep
        if prox_instep > hieghesadst:
            hieghesadst = prox_instep
        if not prox_instep == np.nan:
            hieghessadst += prox_instep
            indeca += 1
    ratios = []
    # on a frame where the ball velocity flips, and prox_instep < prox_knee:
    if np.isfinite(prox_instep):
        ratios.append(prox_instep)
# after the loop:
print(np.mean(ratios))

print(hieghest)
print(hieghesadst)
print(np.divide(hieghessadst,indeca))