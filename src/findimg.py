import cv2
import os
from pathlib import Path

labelled_folder = Path("data/labelled")
video_folder = Path("recordings")
image_folder = Path("data/found_images")

image_folder.mkdir(parents=True, exist_ok=True)

for txt_file in labelled_folder.glob("*.txt"):
    name = txt_file.stem
    print("Processing:", txt_file.name)

    if "_" not in name: 
        print(f"Filename {name} does not contain an underscore. Skipping.")
        continue

    video_name, frame_number = name.rsplit("_", 1)

    frame_number = int(frame_number)

    video_path = video_folder / video_name

    if not video_path.exists():
        for ext in [".mp4", ".MP4", ".mov", ".MOV"]:
            potential_path = video_folder / f"{video_name}{ext}"

            if potential_path.exists():
                video_path = potential_path
                break

    if not video_path.exists():
        print(f"Video file for {video_name} not found. Skipping.")
        continue

    video = cv2.VideoCapture(str(video_path))

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = video.read()


    if success:
        output_image_path = image_folder / f"{name}.jpg"
        cv2.imwrite(str(output_image_path), frame)
        print(f"Saved frame {frame_number} from {video_name} to {output_image_path}")
    else:
        print(f"Failed to read frame {frame_number} from {video_name}.")

    print("Images in output folder:", list(image_folder.glob("*.jpg")))

    video.release()