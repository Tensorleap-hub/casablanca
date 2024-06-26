import lpips
import numpy as np
import torch
import tensorflow as tf


def turn_to_pytorch_tensor(img):
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
    src_image = turn_to_pytorch_tensor(src_image)
    pred_image = turn_to_pytorch_tensor(pred_image)
    src_image = src_image.permute(0, 3, 1, 2)
    pred_image = pred_image.permute(0, 3, 1, 2)
    loss_fn_alex = lpips.LPIPS(net='alex')
    result = loss_fn_alex(src_image, pred_image)
    return (result[0, 0, 0]).detach()


def lpip_loss_vgg(src_image, pred_image):
    src_image = turn_to_pytorch_tensor(src_image)
    pred_image = turn_to_pytorch_tensor(pred_image)
    src_image = src_image.permute(0, 3, 1, 2)
    pred_image = pred_image.permute(0, 3, 1, 2)
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    result = loss_fn_vgg(src_image, pred_image)
    return (result[0, 0, 0]).detach()


def dummy_loss(src_image, pred_image):
    src_image = turn_to_pytorch_tensor(src_image)
    pred_image = turn_to_pytorch_tensor(pred_image)
    return torch.tensor([0])
