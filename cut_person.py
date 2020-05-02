import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage

# == Parameters
# =======================================================================
BLUR = 21
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 100
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0, 0.0, 1.0)  # In BGR format


def try1():
    # == Processing
    # =======================================================================

    # -- Read image
    # -----------------------------------------------------------------------
    img = cv2.imread('test.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -- Edge detection
    # -------------------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # -- Find contours in edges, sort by area
    # ---------------------------------------------
    contour_info = []
    # _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Previously, for a previous version of cv2, this line was:
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Thanks to notes from commenters, I've updated the code but left this note
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    # -- Create empty mask, draw filled polygon on it corresponding to largest contour
    # ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    # -- Smooth mask, then blur it
    # --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask

    # -- Blend masked img into MASK_COLOR background
    # --------------------------------------
    mask_stack = mask_stack.astype('float32') / 255.0  # Use float matrices,
    img = img.astype('float32') / 255.0  # for easy blending

    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)  # Blend
    masked = (masked * 255).astype('uint8')  # Convert back to 8-bit

    cv2.imshow('img', masked)  # Display
    cv2.waitKey()

    # cv2.imwrite('C:/Temp/person-masked.jpg', masked)           # Save


def second_try():
    img = cv2.imread('test.jpg')
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    height, width, channels = img.shape

    # This is rect = (start_x, start_y, width, height)
    # rect = (200, 0, 600, 600)

    s = (img.shape[0] / 10, img.shape[1] / 10)
    rect = (int(s[0]), int(s[1]), int(img.shape[0] - 2 * s[0]), int(img.shape[1] - 2 *
                                                                    s[1]))

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    plt.imshow(img)
    plt.colorbar()
    plt.show()


def third_try():
    img = cv2.imread('test.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    learning_rate = 100
    mask = cv2.createBackgroundSubtractorMOG2().apply(img, learning_rate)
    temp_mask = np.transpose([] for v in mask)
    color_mask = np.transpose([temp_mask, temp_mask, temp_mask])

    plt.imshow(mask & gray)
    plt.show()
    cv2.imshow('Output', color_mask & img)


def fourth_try():
    img = cv2.imread('test.jpg')

    # resize
    side = 600
    ratio = float(side) / max(img.shape)
    img = skimage.img_as_ubyte(
        skimage.transform.resize(
            img, (int(img.shape[0] * ratio), int(img.shape[1] * ratio))))

    s = (img.shape[0] / 10, img.shape[1] / 10)
    rect = (int(s[0]), int(s[1]), int(img.shape[0] - 2 * s[0]), int(img.shape[1] - 2 *
                                                                    s[1]))

    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    plt.imshow(img)
    plt.colorbar()
    plt.show()


def put_on_background(person1, person2, bg_path):
    # -- Read image
    # -----------------------------------------------------------------------
    p1_img = cv2.imread(person1)
    p2_img = cv2.imread(person2)
    bg_img = cv2.imread(bg_path)

    p1_mask = get_mask(p1_img)
    p2_mask = get_mask(p2_img)

    bg_mask_stack = np.zeros(bg_img.shape)
    
    # -- Blend masked img into MASK_COLOR background
    # --------------------------------------
    mask_stack = p1_mask.astype('float32') / 255.0  # Use float matrices,
    p1_img = p1_img.astype('float32') / 255.0  # for easy blending
    bg_img = bg_img.astype('float32') / 255.0

    bg_mask_stack = np.zeros(bg_img.shape)
    # bg_mask_stack = np.dstack([bg_mask] * 3)
    bg_mask_stack = bg_mask_stack.astype('float32') / 255.0

    p1_img = resize(p1_img, bg_img)
    mask_stack = resize(mask_stack, bg_img)

    new_bg_mask = cut_into(mask_stack, bg_mask_stack)
    new_img = cut_into(p1_img, new_bg_mask)

    masked = (new_bg_mask * new_img) + ((1 - new_bg_mask) * bg_img)  # Blend
    masked = (masked * 255).astype('uint8')  # Convert back to 8-bit

    return masked


def get_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -- Edge detection
    # -------------------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # -- Find contours in edges, sort by area
    # ---------------------------------------------
    contour_info = []
    # _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Previously, for a previous version of cv2, this line was:
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Thanks to notes from commenters, I've updated the code but left this note
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    # -- Create empty mask, draw filled polygon on it corresponding to largest contour
    # ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    # -- Smooth mask, then blur it
    # --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask
    return mask_stack


def cut_out_by_contour(cnt, img):
    # cut the body
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

    img = img[topmost[1]:bottommost[1], leftmost[2]:rightmost[2]]
    return img


def resize(img_to_resize, img2):
    h1, w1, _ = img_to_resize.shape
    h2, w2, _ = img2.shape

    factor = min(h2 / h1, w2 / w1)
    resized = cv2.resize(img_to_resize, (0, 0), fx=factor, fy=factor)
    return resized


def cut_into(s_img, l_img):
    x_offset = 0
    y_offset = 0
    copy = l_img.copy()
    copy[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1]] = s_img

    return copy


if __name__ == "__main__":
    img = put_on_background('test.jpg', 'test.jpg', 'background_fr.jpg')
    cv2.imshow('img', img)
    cv2.waitKey()
