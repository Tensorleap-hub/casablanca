from code_loader.contract.visualizer_classes import LeapImage
import numpy as np


def Image_change_last(image: np.ndarray) -> LeapImage:
    img_np_transposed = np.transpose(image, (1, 2, 0))
    return LeapImage(img_np_transposed.astype(np.float32))


def grid_frames(first_frame: np.ndarray, current_frame: np.ndarray) -> LeapImage:
    # first_frame_transposed = np.transpose(first_frame, (1, 2, 0))
    # current_frame_transposed = np.transpose(current_frame, (1, 2, 0))
    concatenated_image = np.hstack((first_frame, current_frame))

    return LeapImage(concatenated_image.astype(np.float32))
