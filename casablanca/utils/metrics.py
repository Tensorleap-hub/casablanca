from casablanca.utils.loss import turn_to_pytorch_tensor
import lpips

import tensorflow as tf


def lpip_alex_metric(src_image, pred_image):
    src_image = turn_to_pytorch_tensor(src_image)
    pred_image = turn_to_pytorch_tensor(pred_image)
    src_image = src_image.permute(0, 3, 1, 2)
    pred_image = pred_image.permute(0, 3, 1, 2)
    loss_fn_alex = lpips.LPIPS(net='alex')
    result = loss_fn_alex(src_image, pred_image)
    result = result.detach()
    result = result.numpy()
    result = tf.convert_to_tensor(result)
    return result[0, 0, 0]


def lpip_vgg_metric(src_image, pred_image):
    src_image = turn_to_pytorch_tensor(src_image)
    pred_image = turn_to_pytorch_tensor(pred_image)
    src_image = src_image.permute(0, 3, 1, 2)
    pred_image = pred_image.permute(0, 3, 1, 2)
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    result = loss_fn_vgg(src_image, pred_image)
    result = result.detach()
    result = result.numpy()
    result = tf.convert_to_tensor(result)
    return result[0, 0, 0]


def l1(real_image: tf.Tensor, pred_image: tf.Tensor):
    """ from paper: L1 represents the mean absolute pixel difference between reconstructed and real videos.
    - since our src image is cross-video ignore these results (valid for same-identity) (current frame and pre image)
    - in our case, calculate also src image with pred image """

    # Calculate the absolute differences
    abs_difference = tf.abs(pred_image - real_image)

    # Compute the mean over all pixels in the video
    l1_loss = tf.reduce_mean(abs_difference, (1, 2, 3))
    return l1_loss
