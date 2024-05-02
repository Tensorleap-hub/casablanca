from casablanca.data.save_frames_to_gcs import save_frames_to_gcs
from casablanca.utils.packages import install_all_packages

# install_all_packages()

from typing import List, Dict
import cv2
import numpy as np

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import LeapDataType, MetricDirection

from casablanca.data.preprocess import load_data
from casablanca.utils.loss import dummy_loss
from casablanca.utils.metrics import lpip_alex_metric, lpip_vgg_metric, l1
from casablanca.utils.visuelizers import Image_change_last, grid_frames, grid_all
from casablanca.config import CONFIG
from casablanca.utils.general_utils import input_encoder


# Preprocess Function
def preprocess_func() -> List[PreprocessResponse]:
    data = save_frames_to_gcs()

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

def calc_metadata_vals(idx: int, preprocess: PreprocessResponse) -> dict:
    res_dic = {}
    res_dic["idx"] = idx
    keys = ['source_id', 'vid_id', 'source_path', 'vid_path', 'vid_name', 'frame_id', 'source_gender', 'vid_gender']
    for k in keys:
        res_dic[k] = preprocess.data[k].iloc[idx]
    res_dic['frame_id'] = int(res_dic['frame_id'])
    res_dic[
        'vid_frame_combination_id'] = f"{preprocess.data['vid_name'].iloc[idx]}_{preprocess.data['frame_id'].iloc[idx]}"
    res_dic[
        'source_vid_combination_id'] = f"{preprocess.data['source_id'].iloc[idx]}_{preprocess.data['vid_name'].iloc[idx]}"
    res_dic['same_id_source_vid'] = int(preprocess.data["source_id"].iloc[idx] == preprocess.data["vid_id"].iloc[idx])
    return res_dic


def source_image_contrast(frame) -> float:
    img_lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    df = abs(a - b)
    return float(np.mean(df))


def source_image_hsv(frame) -> dict:
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hue_range = np.ptp(hsv_image[:, :, 0])
    saturation_level = np.mean(hsv_image[:, :, 1])

    res = {'hue_range': float(hue_range), 'saturation_level': float(saturation_level)}

    return res


def source_image_lab(frame) -> dict:
    lab_image = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    lightness_mean = np.mean(lab_image[:, :, 0])
    a_mean = np.mean(lab_image[:, :, 1])
    b_mean = np.mean(lab_image[:, :, 2])
    res = {'lightness_mean': float(lightness_mean), 'a_mean': float(a_mean), 'b_mean': float(b_mean)}
    return res


def calc_metadata_stats_func(key: str):
    def calc_metadata_stats(idx: int, preprocess: PreprocessResponse) -> dict:
        res_dic = {}
        if key == "source_image":
            frame = input_encoder_source_image(idx, preprocess)

        elif key == 'current_frame':
            frame = input_encoder_current_frame(idx, preprocess)

        else:
            raise Exception("not supported key for input metadata calculations!")

        r, g, b = cv2.split(frame)
        res_dic.update({"red_mean": float(r.mean()), "green_mean": float(g.mean()), "blue_mean": float(b.mean())})
        res_dic.update({"red_std": float(r.std()), "green_std": float(g.std()), "blue_std": float(b.std())})
        res_dic['brightness'] = float(np.mean(frame))

        res = source_image_hsv(frame)
        res_dic.update(res)
        res = source_image_lab(frame)
        res_dic.update(res)

        res_dic['contrast'] = source_image_contrast(frame)
        return res_dic

    return calc_metadata_stats


leap_binder.set_preprocess(function=preprocess_func)

leap_binder.set_input(function=input_encoder_source_image, name='source_image')
leap_binder.set_input(function=input_encoder_current_frame, name='current_frame')
leap_binder.set_input(function=input_encoder_first_frame, name='first_frame')

leap_binder.set_ground_truth(input_encoder_source_image, 'gt_source_image')

leap_binder.set_metadata(calc_metadata_vals, name='metadata')
leap_binder.set_metadata(calc_metadata_stats_func('source_image'), name='source_image')
leap_binder.set_metadata(calc_metadata_stats_func('current_frame'), name='current_frame')

leap_binder.set_visualizer(Image_change_last, 'Image_change_last', LeapDataType.Image)
leap_binder.set_visualizer(grid_frames, 'grid_frames', LeapDataType.Image)
leap_binder.set_visualizer(grid_all, 'grid_all', LeapDataType.Image)

leap_binder.add_custom_metric(lpip_alex_metric, 'lpip_alex', direction=MetricDirection.Downward)
leap_binder.add_custom_metric(lpip_vgg_metric, 'lpip_vgg', direction=MetricDirection.Downward)
leap_binder.add_custom_metric(l1, 'l1', direction=MetricDirection.Downward)
leap_binder.add_custom_loss(dummy_loss, 'dummy_loss')

if __name__ == '__main__':
    leap_binder.check()
