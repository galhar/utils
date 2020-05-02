import cv2
import imutils
import numpy as np


def mask_img(img, mask, bg=None, is_reverted=False, debug_mode=False):
    """

    :param img: the main image
    :param mask: the mask. should be 0 where the image should be and 255 where the background should. otherwise mark
    "is reverted" as true.
    :param bg: the background to put for the image according to the mask
    :param is_reverted: if the mask is black where the image should be and white where the background should
    :param debug_mode:
    :return:
    """
    thresh = 100
    h, w, _ = img.shape
    resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)

    if is_reverted:
        resized_mask[resized_mask <= thresh] = 1
        resized_mask[resized_mask > thresh] = 0

    else:
        resized_mask[resized_mask <= thresh] = 0
        resized_mask[resized_mask > thresh] = 1

    if debug_mode:
        cv2.imshow('mask', resized_mask * 255)
        cv2.waitKey()

    if bg is None:
        final_img = resized_mask * img

    else:
        resized_bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)

        masked_bg = (1 - resized_mask) * resized_bg

        if debug_mode:
            cv2.imshow('background masked', masked_bg)
            cv2.waitKey()

        final_img = resized_mask * img + masked_bg

    return  final_img


if __name__ == '__main__':
    img_name = "./background_creation/blue.jpg"
    img = cv2.imread(img_name)
    mask = cv2.imread("./background_creation/trophy_mask.png")
    bg = cv2.imread("./background_creation/background.png")

    combined = mask_img(img, mask, bg=bg, is_reverted=True)

    cv2.imshow('combined', combined)
    cv2.imwrite(img_name.split('.jpg')[0] + '_trophy.jpg', combined)
    cv2.waitKey()
