import os
import csv
import cv2
import numpy as np
from textwrap import dedent
from numpy.lib.stride_tricks import sliding_window_view

video_path = "recordings/Home_1.mov"
stem = os.path.splitext(os.path.basename(video_path))[0]

with open(f"outputs/custom_{stem}.csv", newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)

    data = {header: [] for header in reader.fieldnames}
    for row in reader:
        for header in reader.fieldnames:
            data[header].append(row[header])

with np.load(f"outputs/cache_{stem}.npz") as cache:
    pose_lm = cache["pose_lm"]
    fps = float(cache["fps"])
    width = int(cache["width"])
    height = int(cache["height"])

with open("src/custom_overlay.py", encoding='utf-8') as file:
    code = file.read()

cap = cv2.VideoCapture(video_path)

roll = int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) % 360
already_upright = cap.get(cv2.CAP_PROP_ORIENTATION_AUTO) == 1

quarter_turn = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE
}

# same camera setup
start = code.index("prev_gray = None")
end = code.index("\nwith mp_pose.Pose()", start)
exec(code[start:end])

# same camera motion logic
start = code.index("        gray = cv2.cvtColor")
end = code.index("\n        ball_found = False", start)
camera_logic = compile(
    dedent(code[start:end]), "custom_overlay.py", "exec"
)

while True:
    success, frame = cap.read()
    if not success:
        break

    if not already_upright and roll in quarter_turn:
        frame = cv2.rotate(frame, quarter_turn[roll])

    exec(camera_logic)

cap.release()

# same preprocessing + actual juggle logic, stopping before rendering
start = code.index("cam_dx = np.array(cam_dx_list, dtype=float)")
end = code.index(
    '\nstem = os.path.splitext(os.path.basename(video_path))[0]', start
)
exec(code[start:end])