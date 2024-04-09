
import os
from os import environ
import onnxruntime

from leap_binder import input_encoder_image, input_encoder_video, preprocess_func


def check_custom_test():
    if environ.get('AUTH_SECRET') is None:
        print("The AUTH_SECRET system variable must be initialized with the relevant secret to run this test")
        exit(-1)
    print("started custom tests")

    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'models/model.onnx'
    sess = onnxruntime.InferenceSession(os.path.join(dir_path, model_path))

    subsets = preprocess_func()
    train, val = subsets
    responses_set = train

    for idx in range(20):
        print(f'start idx: {idx}')
        img = input_encoder_image(idx, responses_set)
        video = input_encoder_video(idx, responses_set)

    print("successfully!")


if __name__ == "__main__":
    video = input_encoder_video(0, 'train')

    # check_custom_test()




