import lpips
import numpy as np
import torch
import tensorflow as tf


def check_tensor(img):
    ptensor = None
    if isinstance(img, np.ndarray):
        ptensor = torch.from_numpy(img)
    elif tf.is_tensor(img):
        img = tf.stop_gradient(img)
        img_np = img.numpy()
        ptensor = torch.from_numpy(img_np)
    elif isinstance(img, torch.Tensor):
        ptensor = img
    return ptensor.detach()


def lpip_loss_alex(src_image, pred_image):
    src_image = check_tensor(src_image)
    pred_image = check_tensor(pred_image)
    loss_fn_alex = lpips.LPIPS(net='alex')
    result = loss_fn_alex(src_image, pred_image)
    return (result[0, 0, 0]).detach()


def lpip_loss_vgg(src_image, pred_image):
    src_image = check_tensor(src_image)
    pred_image = check_tensor(pred_image)
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    result = loss_fn_vgg(src_image, pred_image)
    return (result[0, 0, 0]).detach()


def dummy_loss(src_image, pred_image):
    print('src_image:')
    src_image = check_tensor(src_image)
    print('pred_image:')

    pred_image = check_tensor(pred_image)
    return torch.tensor([0])
