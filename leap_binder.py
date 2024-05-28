from casablanca.data.save_frames_to_gcs import save_frames_to_gcs
from casablanca.utils.packages import install_all_packages

install_all_packages()

from typing import List, Dict
import cv2
import numpy as np

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import LeapDataType, MetricDirection

from casablanca.data.preprocess import load_data, load_data_all
from casablanca.utils.loss import dummy_loss
from casablanca.utils.metrics import lpip_alex_metric, lpip_vgg_metric, l1
from casablanca.utils.visuelizers import image_change_last, grid_frames, grid_all
from casablanca.config import CONFIG
from casablanca.utils.general_utils import input_encoder


# Preprocess Function
def preprocess_func() -> List[PreprocessResponse]:
    data = load_data_all()
    dev = data[data['subset'] == 'dev']
    test = data[data['subset'] == 'test']

    train_df = save_frames_to_gcs(dev.head(CONFIG['dataset_settings']['more_frames']['train_size']))
    val_df = save_frames_to_gcs(test.head(CONFIG['dataset_settings']['more_frames']['test_size']))

    # train_df = data.sample(frac=CONFIG['train_ratio'], random_state=42)
    # val_df = data.drop(train_df.index)

    train = PreprocessResponse(length=train_df.shape[0], data=train_df)
    val = PreprocessResponse(length=val_df.shape[0], data=val_df)
    response = [train, val]
    return response


# ----------------inputs----------------------
# def input_encoder_source_image(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
#     path = preprocess.data['path'].iloc[idx]
#     frame_number = 0
#     return input_encoder(path, frame_number)

# def pick_a_place(idx):
#     if CONFIG['type'] == 'more_frames':
#         place = idx + (idx/2) + 1
#     else:
#         place = (idx*2) + 1
#
#     return place


def input_encoder_current_frame(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    path = preprocess.data['path'].iloc[idx]
    frame_number = preprocess.data['frame_id'].iloc[idx]
    return input_encoder(path, frame_number)


def input_encoder_first_frame(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    path = preprocess.data['path'].iloc[idx]
    frame_number = 0
    return input_encoder(path, frame_number)


# ----------------metadata----------------------

def calc_metadata_vals(idx: int, preprocess: PreprocessResponse) -> dict:
    res_dic = {"idx": idx}
    keys = ['id', 'path', 'vid_name', 'vid_clip_name', 'frame_id', 'frame_path']
    for k in keys:
        res_dic[k] = preprocess.data[k].iloc[idx]
    res_dic['number_id'] = int(res_dic['id'][2:])
    res_dic['vid_clip_name'] = int(res_dic['vid_clip_name'])
    res_dic['frame_id'] = int(res_dic['frame_id'])
    return res_dic


def image_contrast(frame) -> float:
    img_lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    df = abs(a - b)
    return float(np.mean(df))


def image_hsv(frame) -> dict:
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hue_range = np.ptp(hsv_image[:, :, 0])
    saturation_level = np.mean(hsv_image[:, :, 1])

    res = {'hue_range': float(hue_range), 'saturation_level': float(saturation_level)}

    return res


def image_lab(frame) -> dict:
    lab_image = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    lightness_mean = np.mean(lab_image[:, :, 0])
    a_mean = np.mean(lab_image[:, :, 1])
    b_mean = np.mean(lab_image[:, :, 2])
    res = {'lightness_mean': float(lightness_mean), 'a_mean': float(a_mean), 'b_mean': float(b_mean)}
    return res


def convert_to_grayscale(frame):
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def laplacian_var(frame) -> dict:
    frame = convert_to_grayscale(frame)
    laplacian_var = cv2.Laplacian(frame.astype(np.float64), cv2.CV_64F).var()
    res = {'laplacian_var': float(np.round(laplacian_var, 3))}
    return res


def calc_metadata_stats_func(key: str):
    def calc_metadata_stats(idx: int, preprocess: PreprocessResponse) -> dict:
        res_dic = {}
        if key == "source_image":
            frame = input_encoder_first_frame(idx, preprocess)

        elif key == 'current_frame':
            frame = input_encoder_current_frame(idx, preprocess)

        else:
            raise Exception("not supported key for input metadata calculations!")

        r, g, b = cv2.split(frame)
        res_dic.update({"red_mean": float(r.mean()), "green_mean": float(g.mean()), "blue_mean": float(b.mean())})
        res_dic.update({"red_std": float(r.std()), "green_std": float(g.std()), "blue_std": float(b.std())})
        res_dic['brightness'] = float(np.mean(frame))

        res = image_hsv(frame)
        res_dic.update(res)
        res = image_lab(frame)
        res_dic.update(res)
        res = laplacian_var(frame)
        res_dic.update(res)

        res_dic['contrast'] = image_contrast(frame)
        return res_dic

    return calc_metadata_stats


leap_binder.set_preprocess(function=preprocess_func)

leap_binder.set_input(function=input_encoder_first_frame, name='source_image')
leap_binder.set_input(function=input_encoder_current_frame, name='current_frame')
leap_binder.set_input(function=input_encoder_first_frame, name='first_frame')

leap_binder.set_ground_truth(input_encoder_current_frame, 'gt_current_frame')

leap_binder.set_metadata(calc_metadata_vals, name='metadata')
leap_binder.set_metadata(calc_metadata_stats_func('source_image'), name='source_image')
leap_binder.set_metadata(calc_metadata_stats_func('current_frame'), name='current_frame')

leap_binder.set_visualizer(image_change_last, 'Image_change_last', LeapDataType.Image)
leap_binder.set_visualizer(grid_frames, 'grid_frames', LeapDataType.Image)
leap_binder.set_visualizer(grid_all, 'grid_all', LeapDataType.Image)

leap_binder.add_custom_metric(lpip_alex_metric, 'lpip_alex', direction=MetricDirection.Downward)
leap_binder.add_custom_metric(lpip_vgg_metric, 'lpip_vgg', direction=MetricDirection.Downward)
leap_binder.add_custom_metric(l1, 'l1', direction=MetricDirection.Downward)
leap_binder.add_custom_loss(dummy_loss, 'dummy_loss')

if __name__ == '__main__':
    leap_binder.check()
