import os
import cv2
from tqdm import tqdm
from pathlib import Path

video_dir = Path("data")
source_dir = Path("frames_before")

for video_folder_path in tqdm(sorted(video_dir.iterdir())):
    folder_name = video_folder_path.name
    folder_name = folder_name.replace(".mp4", "")
    new_dir_path = source_dir / folder_name
    new_dir_path.mkdir(parents=True, exist_ok=True)
    video_id = video_folder_path.name
    if "vtv" not in video_id:
        continue
    video_path = video_folder_path  # Full path to the video file
    cap = cv2.VideoCapture(str(video_path))  # Convert to string
    frame_number = 0
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 5 == 0:
            # Save frame as an image in the output folder
            frame_filename = new_dir_path / f"frame_{frame_count:06d}.png"
            cv2.imwrite(str(frame_filename), frame)  # Convert to string
            frame_number += 1
        frame_count += 1
    print(frame_number)
    cap.release()
    print("Frames saved to", folder_name)
