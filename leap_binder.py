
from casablanca.utils.packages import install_all_packages

# install_all_packages()

import random
from typing import List, Dict, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.io import read_video

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import LeapDataType, MetricDirection

from casablanca.data.preprocess import load_data
from casablanca.utils.gcs_utils import _download
from casablanca.utils.loss import lpip_loss_alex, lpip_loss_vgg, dummy_loss
from casablanca.utils.metrics import lpip_alex_metric, lpip_vgg_metric
from casablanca.utils.visuelizers import Image_change_last, grid_frames
from casablanca.config import CONFIG


# Preprocess Function
def preprocess_func() -> List[PreprocessResponse]:
    data = load_data()

    train_df = data.sample(frac=CONFIG['train_ratio'], random_state=42)
    val_df = data.drop(train_df.index)

    train = PreprocessResponse(length=train_df.shape[0], data=train_df)
    val = PreprocessResponse(length=val_df.shape[0], data=val_df)
    response = [train, val]
    return response


# def input_encoder_source_image(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
#     filename = preprocess.data['videos'][idx]
#     img = Image.open(filename).convert('RGB')
#     img = img.resize((256, 256))
#     img = np.asarray(img)
#     img = np.transpose(img, (2, 0, 1)) / 255.0
#     img = (img - 0.5) * 2.0
#     img = img[np.newaxis, ...]
#     return torch.tensor(img, dtype=torch.float32)

def input_video(video_name, frame_number) -> np.ndarray:
    fpath = _download(str(video_name))
    vid_dict = read_video(fpath, pts_unit='sec')
    vid = vid_dict[0].permute(0, 3, 1, 2)
    if vid.shape[2] != 256:
        vid = nn.functional.interpolate(vid.to(dtype=torch.float32), size=(256, 256), mode='bilinear',
                                        align_corners=False)
    vid = vid.unsqueeze(0)
    vid_norm = (vid / 255.0 - 0.5) * 2.0
    vid_norm = vid_norm[0]

    return vid_norm[frame_number]


def input_encoder_video(idx: int, preprocess: PreprocessResponse, frame_number) -> np.ndarray:
    filename = preprocess.data['videos'][idx]
    vid = input_video(filename, frame_number)

    return vid


def get_video_name(idx: int, preprocess: PreprocessResponse) -> str:
    filenames = preprocess.data['videos']
    filename = preprocess.data['videos'][idx]
    filename_id = filename.split('/')[3]
    if idx % 2 == 0:
        same_ids = [file for file in filenames if file.split('/')[3] == filename_id and file != filename]
        if not same_ids:
            video_name = filename
        else:
            random.seed(42)
            video_name = random.choice(same_ids)

    else:
        diff_ids = [file for file in filenames if file.split('/')[3] != filename_id]
        if not diff_ids:
            video_name = filename
        else:
            random.seed(42)
            video_name = random.choice(diff_ids)
    return video_name


def input_encoder_source_image(idx: int, preprocess: PreprocessResponse):
    video_name = get_video_name(idx, preprocess)
    print(f'video_name:{video_name}')
    frame_number = 1
    frame = input_video(video_name, frame_number)
    frame = frame.numpy().astype(np.float32)
    return np.transpose(frame, (1, 2, 0))

    # return frame.numpy().astype(np.float32)


def get_id_of_source_image(idx: int, preprocess: PreprocessResponse) -> str:
    video_name = get_video_name(idx, preprocess)
    return video_name.split('/')[3]


def get_utterances_of_source_image(idx: int, preprocess: PreprocessResponse) -> str:
    video_name = get_video_name(idx, preprocess)
    return video_name.split('/')[-3] + '/' + video_name.split('/')[-2]


def get_video_path_of_source_image(idx: int, preprocess: PreprocessResponse) -> str:
    video_name = get_video_name(idx, preprocess)
    return video_name.split('/')[-3] + '/' + video_name.split('/')[-2] + '/' + video_name.split('/')[-1]


def input_encoder_current_frame(idx: int, preprocess: PreprocessResponse):
    frame_number = 10
    frame = input_encoder_video(idx, preprocess, frame_number)
    frame.numpy().astype(np.float32)
    return np.transpose(frame, (1, 2, 0))

    # return frame.numpy().astype(np.float32)


def input_encoder_first_frame(idx: int, preprocess: PreprocessResponse):
    frame_number = 0
    frame = input_encoder_video(idx, preprocess, frame_number)
    frame.numpy().astype(np.float32)
    return np.transpose(frame, (1, 2, 0))

    # return frame.numpy().astype(np.float32)


def get_idx(idx: int, preprocess: PreprocessResponse) -> int:
    return idx


def get_fname(idx: int, preprocess: PreprocessResponse) -> str:
    path = preprocess.data['videos'][idx]
    return path.split('/')[-3] + '/' + path.split('/')[-2] + '/' + path.split('/')[-1]


def get_folder_name(idx: int, preprocess: PreprocessResponse) -> str:
    path = preprocess.data['videos'][idx]
    return path.split('/')[-3] + '/' + path.split('/')[-2]


def get_id(idx: int, preprocess: PreprocessResponse) -> str:
    path = preprocess.data['videos'][idx]
    return path.split('/')[-3]


def get_original_width(video) -> int:
    return int(video.shape[2])


def get_original_height(video) -> int:
    return int(video.shape[3])


def source_image_brightness(frame) -> float:
    return float(np.mean(frame))


def source_image_color_brightness_mean(idx: int, preprocess: PreprocessResponse) -> dict:
    frame = (input__image(idx, preprocess))
    frame = np.transpose(frame, (1, 2, 0))
    b, g, r = cv2.split(frame)
    res = {"red": float(r.mean()), "green": float(g.mean()), "blue": float(b.mean())}

    return res


def source_image_color_brightness_std(idx: int, preprocess: PreprocessResponse) -> dict:
    frame = (input__image(idx, preprocess))
    frame = np.transpose(frame, (1, 2, 0))
    b, g, r = cv2.split(frame)
    res = {"red": float(r.std()), "green": float(g.std()), "blue": float(b.std())}

    return res


def source_image_contrast(frame) -> float:
    img_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    df = abs(a - b)

    return float(np.mean(df))


def source_image_hsv(idx: int, preprocess: PreprocessResponse) -> Dict[str, float]:
    frame = (input__image(idx, preprocess))
    frame = np.transpose(frame, (1, 2, 0))
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hue_range = np.ptp(hsv_image[:, :, 0])  #
    saturation_level = np.mean(hsv_image[:, :, 1])

    res = {'hue_range': float(hue_range), 'saturation_level': float(saturation_level)}

    return res


def source_image_lab(idx: int, preprocess: PreprocessResponse) -> Dict[str, float]:
    frame = (input_encoder_source_image(idx, preprocess))
    # frame = np.transpose(frame, (1, 2, 0))
    lab_image = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    lightness_mean = np.mean(lab_image[:, :, 0])
    a_mean = np.mean(lab_image[:, :, 1])
    b_mean = np.mean(lab_image[:, :, 2])
    res = {'lightness_mean': float(lightness_mean), 'a_mean': float(a_mean), 'b_mean': float(b_mean)}

    return res


def get_video(idx: int, preprocess: PreprocessResponse):
    filename = preprocess.data['videos'][idx]
    fpath = _download(str(filename))
    vid_dict = read_video(fpath, pts_unit='sec')
    video = vid_dict[0].permute(0, 3, 1, 2)
    return video


def input__image(idx: int, preprocess: PreprocessResponse):
    frame_number = 0
    frame = input_encoder_video(idx, preprocess, frame_number)
    return frame.numpy().astype(np.float32)


def metadata_dict(idx: int, preprocess: PreprocessResponse) -> Dict[str, Union[float, int, str]]:
    frame = (input__image(idx, preprocess))
    frame = np.transpose(frame, (1, 2, 0))
    video = get_video(idx, preprocess)

    metadata = {
        'source_image_brightness': source_image_brightness(frame),
        'source_image_contrast': source_image_contrast(frame),
        'get_original_width': get_original_width(video),
        'get_original_height': get_original_height(video)

    }

    return metadata


leap_binder.set_preprocess(function=preprocess_func)

leap_binder.set_input(function=input_encoder_source_image, name='source_image')
leap_binder.set_input(function=input_encoder_current_frame, name='current_frame')
leap_binder.set_input(function=input_encoder_first_frame, name='first_frame')

leap_binder.set_ground_truth(input_encoder_source_image, 'gt_source_image')

leap_binder.set_metadata(get_idx, name='idx')
leap_binder.set_metadata(get_id, name='id')
leap_binder.set_metadata(get_folder_name, name='utterances')
leap_binder.set_metadata(get_fname, name='video_path')
leap_binder.set_metadata(get_id_of_source_image, name='id_of_source_image')
leap_binder.set_metadata(get_utterances_of_source_image, name='utterances_of_source_image')
leap_binder.set_metadata(get_video_path_of_source_image, name='video_path_of_source_image')
leap_binder.set_metadata(metadata_dict, name='')
leap_binder.set_metadata(source_image_color_brightness_mean, name='source_image_color_brightness_mean')
leap_binder.set_metadata(source_image_color_brightness_std, name='source_image_color_brightness_std')
leap_binder.set_metadata(source_image_hsv, name='source_image_hsv')
leap_binder.set_metadata(source_image_lab, name='source_image_lab')

leap_binder.set_visualizer(Image_change_last, 'Image_change_last', LeapDataType.Image)
leap_binder.set_visualizer(grid_frames, 'grid_frames', LeapDataType.Image)

leap_binder.add_custom_metric(lpip_alex_metric, 'lpip_alex', direction=MetricDirection.Upward)
leap_binder.add_custom_metric(lpip_vgg_metric, 'lpip_vgg', direction=MetricDirection.Upward)
leap_binder.add_custom_loss(dummy_loss, 'dummy_loss')

if __name__ == '__main__':
    leap_binder.check()
