import math

import numpy as np
import cv2 as cv
import typing as tp


def flip_horizontal(img: np.ndarray, *args, **kwargs) -> np.ndarray:
    return cv.flip(img, 1)


def flip_vertical(img: np.ndarray, *args, **kwargs) -> np.ndarray:
    return cv.flip(img, 0)


def color_filter(img: np.ndarray,
                 filter_kernel: np.ndarray,
                 *args,
                 **kwargs
                 ) -> np.ndarray:
    filtered = cv.filter2D(src=img, ddepth=-1, kernel=filter_kernel),
    return filtered[0]


def pillarbox(img: np.ndarray, pillarbox_target_ratio: np.float, *args, **kwargs) -> np.ndarray:
    height, width, *_ = img.shape
    target_width = int(pillarbox_target_ratio * height)
    source_ratio = width / height

    if pillarbox_target_ratio < source_ratio:
        return img

    gauss = cv.getGaussianKernel(20, 10)
    kernel = gauss * gauss.transpose(1, 0)

    blurred = color_filter(img, filter_kernel=kernel)
    background = scale_to_fit(blurred, height, target_width)

    skip_cols_left = (target_width - width) // 2
    skip_cols_right = target_width - width - skip_cols_left

    background[:, skip_cols_left: target_width - skip_cols_right, :] = img[:, :, :]

    return background


def letterbox(img: np.ndarray,
              letterbox_target_ratio: np.float,
              fill: tp.Optional[str] = 'blur',
              *args,
              **kwargs
              ) -> np.ndarray:
    height, width, *_ = img.shape

    target_height = int(letterbox_target_ratio * width)
    source_ratio = width / height
    if letterbox_target_ratio == source_ratio:
        return img

    if fill == 'blur':
        gauss = cv.getGaussianKernel(20, 10)
        kernel = gauss * gauss.transpose(1, 0)

        blurred = color_filter(img, filter_kernel=kernel)
    elif fill == 'white':
        blurred = np.full(img.shape, 255, dtype=np.uint8)
    else:
        blurred = np.zeros_like(img)

    background = scale_to_fit(blurred, target_height, width)

    skip_rows_top = (target_height - height) // 2
    skip_rows_bottom = target_height - height - skip_rows_top

    background[skip_rows_top: target_height - skip_rows_bottom, :, :] = img[:, :, :]

    return background


def windowbox(img: np.ndarray,
              height_to_add: int = 2,
              width_to_add: int = 2,
              fill: str = 'blur',
              *args,
              **kwargs
              ) -> np.ndarray:

    height, width, *_ = img.shape

    rows_top = height_to_add // 2
    rows_bottom = height_to_add - rows_top

    cols_left = width_to_add // 2
    cols_right = width_to_add - cols_left

    if fill == 'blur':
        gauss = cv.getGaussianKernel(20, 10)
        kernel = gauss * gauss.transpose(1, 0)
        blurred = color_filter(img, filter_kernel=kernel)

    elif fill == 'white':
        blurred = np.full(img.shape, 255, dtype=np.uint8)
    else:
        blurred = np.zeros_like(img)

    background = scale_to_fit(blurred, height + height_to_add, width + width_to_add)

    background[rows_top: height + height_to_add - rows_bottom, cols_left: width + width_to_add - cols_right, :] = img[:, :, :]

    return background


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
    return cv.resize(img, (target_width, target_height), interpolation=cv.INTER_LINEAR)
