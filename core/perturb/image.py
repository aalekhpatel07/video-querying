import math
import pdb

import numpy as np
import cv2 as cv
import typing as tp


def flip_horizontal(img: np.ndarray, *args, **kwargs) -> np.ndarray:
    return cv.flip(img, 1)


def flip_vertical(img: np.ndarray, *args, **kwargs) -> np.ndarray:
    return cv.flip(img, 0)


def color_filter(img: np.ndarray, filter_kernel: np.ndarray, *args, **kwargs) -> np.ndarray:
    return cv.erode(img, filter_kernel)


def pillarbox(img: np.ndarray, pillarbox_target_ratio: np.float, *args, **kwargs) -> np.ndarray:
    height, width, *_ = img.shape

    return img


def letterbox(img: np.ndarray, letterbox_target_ratio: np.float, *args, **kwargs) -> np.ndarray:
    return img


def windowbox(img: np.ndarray, windowbox_target_ratio: np.float, *args, **kwargs) -> np.ndarray:
    return img


def rotate(img: np.ndarray, rotation_angle: np.float, *args, **kwargs) -> np.ndarray:
    rotation_angle = rotation_angle * 180 / math.pi
    height, width, *_ = img.shape
    image_center = int(height / 2), int(width / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, rotation_angle, 1.0)
    result = cv.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


def crop(
    img: np.ndarray,
    remove_rows: int,
    remove_cols: int,
    *args,
    **kwargs
) -> np.ndarray:
    height, width, *_ = img.shape

    top_rem = int(remove_rows / 2)
    bottom_rem = remove_rows - top_rem

    left_rem = int(remove_cols / 2)
    right_rem = remove_cols - left_rem

    if remove_rows >= height + 20 or remove_cols >= width + 20:
        return img

    return img[top_rem:height - bottom_rem, left_rem: width - right_rem, :]


def scale_uniform(
    img: np.ndarray,
    scale_ratio: float,
    *args,
    **kwargs
) -> np.ndarray:
    return scale(img, scale_x=scale_ratio, scale_y=scale_ratio)


def scale(
    img: np.ndarray,
    scale_x: float,
    scale_y: float,
    *args,
    **kwargs
) -> np.ndarray:
    height, width, *_ = img.shape
    return cv.resize(img, (0, 0), fx=scale_x, fy=scale_y, interpolation=cv.INTER_LINEAR)


def scale_to_fit(
    img: np.ndarray,
    target_height: int,
    target_width: int,
    *args,
    **kwargs
) -> np.ndarray:
    return cv.resize(img, (target_height, target_width), interpolation=cv.INTER_LINEAR)
