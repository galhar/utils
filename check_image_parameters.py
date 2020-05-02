# Writer: Gal Harari
# Date: 31/12/2019

"""
This file creates a comfortable platform to asses numerical values for image processing.
When you finish press "q"! It saves everything.
You'll need a function that process an image and gets numerical arguments and the img itself. Then you need to create a
dictionary of the arguments' names it gets and their upper bounds (lower bound is set to zero).
A usage example can be seen in "if __name__ == "__main__"" part.
"""


import cv2
import numpy as np
import imutils
import json

IMAGE_STRING = 'img'


def nothing(x):
    pass


def check_params(frame, args_dict, processing_func):
    """
    it loads the values last used ro defining the upper bound as default.
    when you press "q" it ends and saves the values last used.
    :param frame:
    :param args_dict: the keys are the names of the arguments the function gets, the values are tuplesthe upper bound of
     each arg
    :param processing_func: the func the gets an image, the argumnets of the args_dict, and the image into the arg named
     as IMAGE_STRING", ITS IMMPORTANT. and returns a list of images to show
    :return:
    """
    if frame is None:
        print("No image inserted. Check your cv2.imread please.")

    cv2.namedWindow("Trackbars")

    create_trackbars(args_dict)

    chosen_args_dict = args_dict.copy()
    chosen_args_dict[IMAGE_STRING] = frame

    while True:
        for arg_name in chosen_args_dict.keys():
            if arg_name == IMAGE_STRING:
                continue
            chosen_args_dict[arg_name] = cv2.getTrackbarPos(arg_name, "Trackbars")

        ret_images_list = processing_func(**chosen_args_dict)

        for i, img in enumerate(ret_images_list):
            cv2.imshow(f"image {i}", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            del chosen_args_dict[IMAGE_STRING]
            for i, img in enumerate(ret_images_list):
                cv2.imwrite(f"image {i}.jpg", img)
            save_args(chosen_args_dict)
            break

    cv2.destroyAllWindows()


def create_trackbars(args_dict):

    loaded = load_mask(args_dict)
    loaded_args_dict = {arg_name: bound for arg_name, bound in loaded.items()}

    for arg_name, val in loaded_args_dict.items():
        # args_dict[arg_name] is the upper bound
        cv2.createTrackbar(arg_name, "Trackbars", val, args_dict[arg_name], nothing)


def save_args(args_dict):
    with open('data.json', 'w') as fp:
        json.dump(args_dict, fp)


def load_mask(args_dict):
    try:
        with open('data.json', 'r') as fp:
            data = json.load(fp)
    except (IOError, json.decoder.JSONDecodeError) as e:
        data = {arg_name: bound for arg_name, bound in args_dict.items()}
    return data


def example_process_func(img, arg1, arg2):
    return img[:, :, :], img[:, arg1:arg2, :]


if __name__ == '__main__':
    # Documentation on top of the page

    print("When you finish press \"q\" instead of closing the windows manually. "
          "It would save the values and the edited images.")
    img_name = 'example_image.jpg'
    img = cv2.imread(img_name)
    arg1_upper_bound, arg2_upper_bound = 400, 1000
    example_args_dict = {'arg1': arg1_upper_bound, 'arg2': arg2_upper_bound}

    check_params(img, example_args_dict, example_process_func)

    print(load_mask(example_args_dict))
