import time

import cv2
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
    while vid_dict[0].shape[0] < frame_number:
        time.sleep(2)
        print(f'reloading video {fpath}')
        vid_dict = read_video(fpath, pts_unit='sec')
    vid = vid_dict[0].permute(0, 3, 1, 2)
    if vid.shape[2] != 256:
        vid = nn.functional.interpolate(vid.to(dtype=torch.float32), size=(256, 256), mode='bilinear',
                                        align_corners=False)
    # vid = vid.unsqueeze(0)
    vid_norm = (vid / 255.0 - 0.5) * 2.0
    # vid_norm = vid_norm[0]

    return vid_norm[frame_number]


# def input_encoder(path, frame_number):
#     root, extension = os.path.splitext(path)
#     fpath = download(str(root + CONFIG['frame_separator'] + str(frame_number) + '.png'))
#     if fpath.rsplit('.', 1)[-1] == 'mp4':
#         frame = input_video(fpath, frame_number)
#         dir_path = fpath.rsplit('.', 1)[0] + CONFIG['frame_separator'] + str(frame_number) + '.png'
#         frame = rescale_min_max(frame.numpy()).transpose((1, 2, 0))
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         cv2.imwrite(dir_path, frame)
#     else:
#         frame = input_encoder_image(fpath)
#
#     return frame.astype(np.float32)

def input_encoder(path, frame_number):
    root, extension = os.path.splitext(path)
    frame_path = str(root + CONFIG['frame_separator'] + str(frame_number) + '.png')
    fpath = _download(frame_path)
    frame = input_encoder_image(fpath)

    return frame.astype(np.float32)