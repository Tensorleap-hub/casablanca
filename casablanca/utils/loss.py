import lpips
import numpy as np
import torch

def check_tensor(img):
    if isinstance(img, np.ndarray):
        return torch.from_numpy(img)
    else:
        return img


def dummy_loss(src_image, pred_image):
    return torch.tensor([0])


def lpip_loss_alex(src_image, pred_image):
    print(f'src_image shape {src_image.shape}')
    print(f'pred_image shape {pred_image.shape}')

    src_image = check_tensor(src_image)
    pred_image = check_tensor(pred_image)
    loss_fn_alex = lpips.LPIPS(net='alex')
    result = loss_fn_alex(src_image, pred_image)
    return result[0, 0, 0]


def lpip_loss_vgg(src_image, pred_image):
    src_image = check_tensor(src_image)
    pred_image = check_tensor(pred_image)
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    result = loss_fn_vgg(src_image, pred_image)
    return result[0, 0, 0]
