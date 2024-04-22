from typing import List

import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision.io import read_video
import os
from PIL import Image

# Tensorleap imports
from code_loader.utils import rescale_min_max

from casablanca.config import CONFIG
from casablanca.utils.gcs_utils import download, _download


def input_encoder_image(filename) -> np.ndarray:
    img = Image.open(filename).convert('RGB')
    img = img.resize((256, 256))
    img = np.asarray(img)
    img = img / 255.0
    img = (img - 0.5) * 2.0
    return img


def input_video(fpath, frame_number) -> np.ndarray:
    vid_dict = read_video(fpath, pts_unit='sec')
    vid = vid_dict[0].permute(0, 3, 1, 2)
    if vid.shape[2] != 256:
        vid = nn.functional.interpolate(vid.to(dtype=torch.float32), size=(256, 256), mode='bilinear',
                                        align_corners=False)
    vid = vid.unsqueeze(0)
    vid_norm = (vid / 255.0 - 0.5) * 2.0
    vid_norm = vid_norm[0]
    return vid_norm[frame_number]


def input_encoder(path, frame_number):
    frame_local_path, frame_exists = is_frame_exists(path, frame_number)
    vid_local_path = _download(path)
    if frame_exists:
        frame = input_encoder_image(frame_local_path)
    else:
        frame = input_video(vid_local_path, frame_number)
        frame = rescale_min_max(frame.numpy()).transpose((1, 2, 0))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(frame_local_path, frame)

    return frame.astype(np.float32)


def is_frame_exists(path, frame_number):
    root, extension = os.path.splitext(path)
    frame_local_path = os.path.join(CONFIG['local_storage_path'],
                                    root + CONFIG['frame_separator'] + str(frame_number) + '.png')
    return frame_local_path, os.path.exists(frame_local_path)


def download_all_vids(paths: List[str]) -> List[str]:
    local_paths = []
    for path in tqdm(paths):
        local_path = _download(path)
        local_paths.append(local_path)
        # print(f"$$$ downloaded {path}\n"
        #       f"to {local_path}")
    return local_paths


def create_and_save_all_frames(data: pd.DataFrame) -> None:
    vid_paths = download_all_vids(data['vid_path'].unique().tolist())
    # vid_paths = data['vid_path'].apply(lambda path: os.path.join(CONFIG['local_storage_path'], path)).unique().tolist()
    frame_ids = data['frame_id'].unique().tolist() + [0]
    for vid_path in vid_paths:
        for frame_id in frame_ids:
            frame_local_path, frame_exists = is_frame_exists(vid_path, frame_id)
            if not frame_exists:
                frame = input_video(vid_path, frame_id)
                frame = rescale_min_max(frame.numpy()).transpose((1, 2, 0))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(frame_local_path, frame)
