from code_loader.contract.visualizer_classes import LeapImage
from code_loader.utils import rescale_min_max
import numpy as np


def image_change_last(image: np.ndarray) -> LeapImage:
    return LeapImage(image.astype(np.float32))


def grid_frames(first_frame: np.ndarray, current_frame: np.ndarray) -> LeapImage:
    concatenated_image = (np.hstack((rescale_min_max(first_frame), rescale_min_max(current_frame))))
    return LeapImage(concatenated_image)


def grid_all(first_frame: np.ndarray, current_frame: np.ndarray, source_image: np.ndarray,
             pred_image: np.ndarray) -> LeapImage:
    first_row = np.hstack((rescale_min_max(source_image), rescale_min_max(pred_image)))
    sec_row = np.hstack((rescale_min_max(first_frame), rescale_min_max(current_frame)))
    concatenated_image = np.vstack((first_row, sec_row))
    return LeapImage(concatenated_image)
