from typing import List

import numpy as np
import torch
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.io import read_video
from casablanca.config import config
import os

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import Metric, DatasetMetadataType
from code_loader.contract.visualizer_classes import LeapHorizontalBar

from casablanca.utils.gcs_utils import download


# Preprocess Function
def preprocess_func() -> List[PreprocessResponse]:
    train_images = download(config['train_file'])
    with open(train_images, 'r') as f:
        train_list_of_paths = f.read().split("\n")[:-1]

    val_images = download(config['val_file'])
    with open(val_images, 'r') as f:
        val_list_of_paths = f.read().split("\n")[:-1]

    train_size = min(len(train_list_of_paths), config['train_size'])
    val_size = min(len(val_list_of_paths), config['val_size'])

    train_list_of_paths = [os.path.join('train', p) for p in train_list_of_paths[:train_size]]
    val_list_of_paths = [os.path.join('val', p) for p in val_list_of_paths[:val_size]]

    train = PreprocessResponse(length=len(train_list_of_paths), data={'images': train_list_of_paths})
    val = PreprocessResponse(length=len(val_list_of_paths), data={'images': val_list_of_paths})
    response = [train, val]
    return response


def input_encoder_image(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    filename = preprocess.data['samples'][idx]['file_name']
    img = Image.open(filename).convert('RGB')
    img = img.resize((256, 256))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1)) / 255.0
    img = (img - 0.5) * 2.0
    img = img[np.newaxis, ...]
    return torch.tensor(img, dtype=torch.float32).cuda()


def input_encoder_video(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    vid_path = preprocess.data['samples'][idx]['file_name_vid']
    vid_dict = read_video(vid_path, pts_unit='sec')
    vid = vid_dict[0].permute(0, 3, 1, 2)
    if vid.shape[2] != 256:
        vid = nn.functional.interpolate(vid, size=(256, 256), mode='bilinear', align_corners=False)
    vid = vid.unsqueeze(0)
    vid_norm = (vid / 255.0 - 0.5) * 2.0

    return vid_norm.cuda(), vid_dict[2]['video_fps']


# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
def gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    return preprocessing.data['labels'][idx].astype('float32')


# Dataset binding functions to bind the functions above to the `Dataset Instance`.
leap_binder.set_preprocess(function=preprocess_func)
leap_binder.set_input(function=input_encoder_image, name='image')
leap_binder.set_input(function=input_encoder_video, name='video')

leap_binder.set_ground_truth(function=gt_encoder, name='classes')
# leap_binder.add_prediction(name='classes', labels=LABELS)
