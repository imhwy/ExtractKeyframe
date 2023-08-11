import cv2
import h5py
# import clip
import torch
import ffmpeg
import logging
import subprocess
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
import json


logging.basicConfig(
    filename="log/extract_keyframe.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# df_extracted_kf = pd.DataFrame(
#     columns=[
#         "video_id",
#         "msb_shot_id",
#         "frame_id",
#         "kf_filename",
#     ]
# )

extracted_kf_json = {}

# Load CLIP model
# logger.info("Loading CLIP...")
# CLIP_version = "ViT-B/32"
device = "cuda" if torch.cuda.is_available() else "cpu"
# logger.info("Device: %s\tCLIP version: %s" % (device, CLIP_version))
# model, preprocess = clip.load(CLIP_version, device=device)
# logger.info("CLIP is ready!")

# Initiate ORB detector
orb = cv2.ORB_create()
# Since we use ORB, we should use hamming distance
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


# hack for steam.run_async(quiet=True) bug
def _run_async_quiet(stream_spec):
    args = ffmpeg._run.compile(
        stream_spec, "/PATH_Programs/ffmpeg.exe", overwrite_output=False)
    print(args)
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def extract_all_frames(video_filename, output_width, output_height):
    video_stream, err = (
        ffmpeg.input(video_filename)
        .output(

            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(output_width, output_height),
        )
        .run(capture_stdout=True, capture_stderr=True)
    )
    return np.frombuffer(video_stream, np.uint8).reshape(
        [-1, output_height, output_width, 3]
    )


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def save_img(
    video_id,
    shot_id,
    frame_idx,
    img,
    save_img_path,
):
    global df_extracted_kf
    img.save(str(save_img_path))
    df_extracted_kf = pd.concat(
        [
            df_extracted_kf,
            pd.DataFrame(
                {
                    "video_id": [video_id],
                    "msb_shot_id": [-1],
                    "frame_id": [frame_idx],
                    "kf_filename": [save_img_path.name],
                }
            ),
        ],
        ignore_index=True,
        axis=0,
    )


def get_video_stream(video_path, output_width, output_height):
    return _run_async_quiet(
        ffmpeg.input(video_path).output("pipe:", format="rawvideo",
                                        pix_fmt="rgb24", s="{}x{}".format(output_width, output_height),)
    )


# def get_img_feature(img):
#     with torch.no_grad():
#         frame_feature = model.encode_image(
#             preprocess(img).unsqueeze(0).to(device))
#         frame_feature /= frame_feature.norm(dim=-1, keepdim=True)
#         frame_feature = frame_feature.cpu().numpy()[0]
#     return frame_feature


def extract_keyframe(
    video_path,
    save_path,
    msb=None,
    thres=0.75,
    output_width=400,
    output_height=225,
    skip_frame=15,
    video_frames=[],
):
    # last_endframe = msb.iloc[-1, 2]
    video_id = video_path.with_suffix("").name

    # Open video stream buffer
    video_stream = get_video_stream(video_path, output_width, output_height)

    processed_frames = 0
    total_kf = 0

    frame_idx = 0
    keyframe = None
    keyframe_feature = None
    last_frame = None
    last_frame_feature = None

    store_frame_ids = []
    store_features = []

    while True:
        in_bytes = video_stream.stdout.read(output_width * output_height * 3)
        if not in_bytes:
            # End of video
            break

        if frame_idx % skip_frame != 0:
            frame_idx += 1
            continue

        processed_frames += 1

        # Get shotid from dataframe
        # shot_id = msb.index[-1] + 1
        # if frame_idx <= last_endframe:
        #     shot_id = msb[
        #         (msb["startframe"] <= frame_idx) & (
        #             msb["endframe"] >= frame_idx)
        #     ].index.values[0]

        # Get keyframe image name and create folder if not exists
        img_name = f"{frame_idx:06d}.jpg"
        save_img_path = (
            save_path / "keyframes" / str(video_id) / img_name
        )
        save_img_path.parent.mkdir(parents=True, exist_ok=True)

        frame = np.frombuffer(in_bytes, np.uint8).reshape(
            [output_height, output_width, 3]
        )

        # Convert frame to PIL Image
        img = Image.fromarray(frame)

        # # ORB Feature
        cv_img = np.array(img.convert("L"))
        kp = orb.detect(cv_img)
        kp, des = orb.compute(cv_img, kp)
        frame_feature = (kp, des)

        if des is None:
            # Skip frame without any keypoint
            continue

        # Select the first frame as the keyframe
        if keyframe is None:
            keyframe = frame.copy()
            keyframe_feature = (kp, des)
            img.save(str(save_img_path))
            total_kf += 1
            video_frames.append(str(img_name))
        else:
            # Compare similarity of current frame vs last kf and last frame
            matches = matcher.knnMatch(
                last_frame_feature[1], frame_feature[1], k=2)
            good = []
            for pair in matches:
                if len(pair) != 2:
                    continue
                m, n = pair
                if m.distance < 0.7*n.distance:
                    good.append([m])

            # matches_img = cv2.drawMatchesKnn(
            #     np.array(Image.fromarray(last_frame).convert("L")),
            #     last_frame_feature[0],
            #     cv_img,
            #     frame_feature[0],
            #     good,
            #     None,
            #     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            # )
            # cv2.imwrite(str(save_img_path), matches_img)

            last_frame_similar = 1 - \
                len(good) / \
                max(len(frame_feature[0]), len(last_frame_feature[0]))

            matches = matcher.knnMatch(
                keyframe_feature[1], frame_feature[1], k=2)
            good = []
            for pair in matches:
                if len(pair) != 2:
                    continue
                m, n = pair
                if m.distance < 0.7*n.distance:
                    good.append([m])
            kf_similar = 1 - \
                len(good) / \
                max(len(frame_feature[0]), len(keyframe_feature[0]))

            if kf_similar > thres or last_frame_similar > thres:
                keyframe = frame.copy()
                keyframe_feature = frame_feature
                img.save(str(save_img_path))
                total_kf += 1
                video_frames.append(str(img_name))

        # Update last frame
        last_frame = frame.copy()
        last_frame_feature = (kp, des)

        # Compure CLIP image feature
        # frame_feature = get_img_feature(img)

        # Select the first frame as the keyframe
        # if keyframe is None:
        #     keyframe = frame.copy()
        #     keyframe_feature = frame_feature.copy()
        #     store_frame_ids.append(frame_idx)
        #     store_features.append(keyframe_feature)
        #     save_img(
        #         video_id=video_id,
        #         shot_id=shot_id,
        #         frame_idx=frame_idx,
        #         img=img,
        #         save_img_path=save_img_path,
        #     )
        #     total_kf += 1
        # else:
        #     # Compare similarity of current frame vs last kf and last frame
        #     kf_similar = cosine_similarity(keyframe_feature, frame_feature)
        #     last_frame_similar = cosine_similarity(
        #         last_frame_feature, frame_feature)

        #     # If the difference is large enough, current frame is the new keyframe
        #     if kf_similar < thres or last_frame_similar < thres:
        #         keyframe = frame.copy()
        #         keyframe_feature = frame_feature.copy()
        #         store_frame_ids.append(frame_idx)
        #         store_features.append(keyframe_feature)
        #         save_img(
        #             video_id=video_id,
        #             shot_id=shot_id,
        #             frame_idx=frame_idx,
        #             img=img,
        #             save_img_path=save_img_path,
        #         )
        #         total_kf += 1

        # # Update last frame
        # last_frame = frame.copy()
        # last_frame_feature = frame_feature.copy()

        frame_idx += 1

    # Open h5py file to store image's features
    # hf_path = save_path / "features" / (video_id + ".h5")
    # hf_path.parent.mkdir(parents=True, exist_ok=True)
    # hf = h5py.File(str(hf_path), "w")
    # hf.create_dataset("frame_ids", data=store_frame_ids)
    # hf.create_dataset("features", data=store_features)
    # hf.close()

    # logger.info("Total video frame: %s" % frame_idx)
    # logger.info("Processed frame: %s" % processed_frames)
    # logger.info("Keyframe extracted: %s" % total_kf)

    return frame_idx, processed_frames, total_kf, video_frames


def find_video_file(basePath, video_id, supported_exts):
    for ext in supported_exts:
        video_path = basePath / (video_id + ext)
        if video_path.exists():
            return video_path


def main():
    supported_exts = [".mp4", ".avi", ".m4v", ".mov", ".mpe", ".vtt"]

    dataset_name = "data-batch-test-test"
    dataset_dir = Path("frames_after") / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    video_dir = Path("data")
    # msb_dir = dataset_dir / "msb"
    save_dir = Path("frames_after") / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    processed_frames = 0
    total_kf = 0

    for video_folder_path in tqdm(sorted(video_dir.iterdir())):
        video_frames = []
        video_id = video_folder_path.name
        if "vtv" not in video_id:
            continue
        print(video_id)
        logger.info("Processing video %s" % video_id)
        video_id = video_id.replace('.mp4', '')
        # if 'C00' not in video_id :
        #     continue
        # if video_id < 'C01' or ('C02_V03' in video_id or 'C02_V02' in video_id):
        #     continue
        # if 'C02_V03' not in video_id :
        #     continue
        video_path = find_video_file(
            video_dir, video_id, supported_exts)
        if not video_path:
            logger.error("Video %s not found. Skip it" % video_id)
            continue

        # msb_path = msb_dir / (video_id + ".tsv")

        # msb = pd.read_csv(str(msb_path), delimiter="\t")

        n_frames, n_procf, n_kf, video_frames = extract_keyframe(
            video_path, save_dir, video_frames)
        total_frames += n_frames
        processed_frames += n_procf
        total_kf += n_kf
        filenames_without_extension = [video_frame[:-4]
                                       for video_frame in video_frames]
        extracted_kf_json[video_id] = filenames_without_extension

    df_extracted_kf.to_csv(str(save_dir / f"{dataset_name}_extracted_kf.csv"))
    logger.info("Total frame: %s" % total_frames)
    logger.info("Total processed frame: %s" % processed_frames)
    logger.info("Total keyframe extracted: %s" % total_kf)

    with open('data.json', 'w') as json_file:
        json.dump(extracted_kf_json, json_file, indent=4)


if __name__ == "__main__":
    main()
