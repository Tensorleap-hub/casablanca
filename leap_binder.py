from casablanca.utils.packages import install_all_packages

install_all_packages()

from typing import List, Dict,
import cv2
import numpy as np


# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import LeapDataType, MetricDirection

from casablanca.data.preprocess import load_data
from casablanca.utils.loss import dummy_loss
from casablanca.utils.metrics import lpip_alex_metric, lpip_vgg_metric
from casablanca.utils.visuelizers import Image_change_last, grid_frames
from casablanca.config import CONFIG
from casablanca.utils.general_utils import input_encoder


# Preprocess Function
def preprocess_func() -> List[PreprocessResponse]:
    data = load_data()

    train_df = data.sample(frac=CONFIG['train_ratio'], random_state=42)
    val_df = data.drop(train_df.index)

    train = PreprocessResponse(length=train_df.shape[0], data=train_df)
    val = PreprocessResponse(length=val_df.shape[0], data=val_df)
    response = [train, val]
    return response


# ----------------inputs----------------------
def input_encoder_source_image(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    path = preprocess.data['source_path'].iloc[idx]
    frame_number = 0
    return input_encoder(path, frame_number)


def input_encoder_current_frame(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    path = preprocess.data['vid_path'].iloc[idx]
    frame_number = preprocess.data['frame_id'].iloc[idx]
    return input_encoder(path, frame_number)


def input_encoder_first_frame(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    path = preprocess.data['vid_path'].iloc[idx]
    frame_number = 0
    return input_encoder(path, frame_number)


# ----------------metadata----------------------
def get_video_name(idx: int, preprocess: PreprocessResponse) -> str:
    return preprocess.data['vid_name'].iloc[idx]


def get_id_of_source_image(idx: int, preprocess: PreprocessResponse) -> str:
    return preprocess.data['source_id'].iloc[idx]


def get_frame_index(idx: int, preprocess: PreprocessResponse) -> str:
    return preprocess.data['frame_id'].iloc[idx]


def get_video_path_of_source_image(idx: int, preprocess: PreprocessResponse) -> str:
    return preprocess.data['source_path'].iloc[idx]


def get_video_path_of_current_frame(idx: int, preprocess: PreprocessResponse) -> str:
    return preprocess.data['vid_path'].iloc[idx]


def get_source_vid_combination_id(idx: int, preprocess: PreprocessResponse) -> str:
    return f"{preprocess.data['source_id'].iloc[idx]}_{preprocess.data['vid_name'].iloc[idx]}"


def get_vid_frame_combination_id(idx: int, preprocess: PreprocessResponse) -> str:
    return f"{preprocess.data['vid_name'].iloc[idx]}_{preprocess.data['frame_id'].iloc[idx]}"


def get_idx(idx: int, preprocess: PreprocessResponse) -> int:
    return idx


def source_image_brightness(frame) -> float:
    return float(np.mean(frame))


def source_image_color_brightness_mean(idx: int, preprocess: PreprocessResponse) -> dict:
    frame = input_encoder_source_image(idx, preprocess)
    r, g, b = cv2.split(frame)
    res = {"red": float(r.mean()), "green": float(g.mean()), "blue": float(b.mean())}

    return res


def source_image_color_brightness_std(idx: int, preprocess: PreprocessResponse) -> dict:
    frame = input_encoder_source_image(idx, preprocess)
    r, g, b = cv2.split(frame)
    res = {"red": float(r.std()), "green": float(g.std()), "blue": float(b.std())}

    return res


def source_image_contrast(frame) -> float:
    img_lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    df = abs(a - b)

    return float(np.mean(df))


def source_image_hsv(idx: int, preprocess: PreprocessResponse) -> Dict[str, float]:
    frame = input_encoder_source_image(idx, preprocess)
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hue_range = np.ptp(hsv_image[:, :, 0])
    saturation_level = np.mean(hsv_image[:, :, 1])

    res = {'hue_range': float(hue_range), 'saturation_level': float(saturation_level)}

    return res


def source_image_lab(idx: int, preprocess: PreprocessResponse) -> Dict[str, float]:
    frame = input_encoder_source_image(idx, preprocess)
    lab_image = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    lightness_mean = np.mean(lab_image[:, :, 0])
    a_mean = np.mean(lab_image[:, :, 1])
    b_mean = np.mean(lab_image[:, :, 2])
    res = {'lightness_mean': float(lightness_mean), 'a_mean': float(a_mean), 'b_mean': float(b_mean)}

    return res


leap_binder.set_preprocess(function=preprocess_func)

leap_binder.set_input(function=input_encoder_source_image, name='source_image')
leap_binder.set_input(function=input_encoder_current_frame, name='current_frame')
leap_binder.set_input(function=input_encoder_first_frame, name='first_frame')

leap_binder.set_ground_truth(input_encoder_source_image, 'gt_source_image')

leap_binder.set_metadata(get_idx, name='idx')
leap_binder.set_metadata(get_video_name, name='video_name')
leap_binder.set_metadata(get_id_of_source_image, name='source_id')
leap_binder.set_metadata(get_frame_index, name='frame_index')
leap_binder.set_metadata(get_video_path_of_source_image, name='video_path_of_source_image')
leap_binder.set_metadata(get_video_path_of_current_frame, name='video_path_of_current_frame')
leap_binder.set_metadata(get_source_vid_combination_id, name='source_vid_combination_id')
leap_binder.set_metadata(get_vid_frame_combination_id, name='vid_frame_combination_id')
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
