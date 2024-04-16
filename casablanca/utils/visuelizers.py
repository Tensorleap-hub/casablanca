from code_loader.contract.visualizer_classes import LeapImage
import numpy as np


def Image_change_last(image: np.ndarray) -> LeapImage:
    img_np_transposed = np.transpose(image, (1, 2, 0))
    return LeapImage(img_np_transposed.astype(np.float32),)
