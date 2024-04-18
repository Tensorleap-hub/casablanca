
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
