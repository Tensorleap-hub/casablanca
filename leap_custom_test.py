import os
from os import environ
import onnxruntime
import numpy as np

from casablanca.utils.loss import lpip_loss_alex, lpip_loss_vgg
from casablanca.utils.visuelizers import Image_change_last
from leap_binder import input_encoder_source_image, input_encoder_video, preprocess_func, input_encoder_current_frame, \
    input_encoder_first_frame, metadata_dict, get_fname, get_folder_name, source_image_color_brightness_mean, \
    source_image_color_brightness_std, source_image_hsv, source_image_lab, get_idx


def check_custom_test():
    if environ.get('AUTH_SECRET') is None:
        print("The AUTH_SECRET system variable must be initialized with the relevant secret to run this test")
        exit(-1)
    print("started custom tests")

    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'model/lia_encoder_decoder.onnx'
    sess = onnxruntime.InferenceSession(os.path.join(dir_path, model_path))

    input_name_1 = sess.get_inputs()[0].name
    input_name_2 = sess.get_inputs()[1].name
    input_name_3 = sess.get_inputs()[2].name
    output_names = [output.name for output in sess.get_outputs()]

    subsets = preprocess_func()
    responses_set = subsets[0]

    for idx in range(10):
        print(f'start idx: {idx}')
        source_image = input_encoder_source_image(idx, responses_set)
        current_frame = input_encoder_current_frame(idx, responses_set)
        first_frame = input_encoder_first_frame(idx, responses_set)

        pred = sess.run(output_names, {input_name_1: np.expand_dims(source_image, 0),
                                       input_name_2: np.expand_dims(first_frame, 0),
                                       input_name_3: np.expand_dims(current_frame, 0)})[0]

        loss_alex = lpip_loss_alex(np.expand_dims(source_image, 0), pred)
        loss_vgg = lpip_loss_vgg(np.expand_dims(source_image, 0), pred)

        source_image_vis = Image_change_last(source_image)
        current_frame_vis = Image_change_last(current_frame)
        first_frame_vis = Image_change_last(first_frame)


        metadata = metadata_dict(idx, responses_set)
        idx_ = get_idx(idx, responses_set)
        file_name = get_fname(idx, responses_set)
        folder_name = get_folder_name(idx, responses_set)
        source_image_color_brightness_mean_ = source_image_color_brightness_mean(idx, responses_set)
        source_image_color_brightness_std_ = source_image_color_brightness_std(idx, responses_set)
        source_image_hsv_ = source_image_hsv(idx, responses_set)
        source_image_lab_ = source_image_lab(idx, responses_set)



    print("successfully!")


if __name__ == "__main__":
    check_custom_test()
