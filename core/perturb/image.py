import numpy as np
import cv2 as cv
import typing as tp


def flip_horizontal(img: np.ndarray, *args, **kwargs) -> np.ndarray:
    return cv.flip(img, 1)


def flip_vertical(img: np.ndarray, *args, **kwargs) -> np.ndarray:
    return cv.flip(img, 0)


def color_filter(img: np.ndarray, filter_: np.ndarray, *args, **kwargs) -> np.ndarray:
    return cv.erode(img, filter_)


def pillarbox(img: np.ndarray, target_ratio: np.float, *args, **kwargs) -> np.ndarray:
    height, width, *_ = img.shape

    return


def letterbox(img: np.ndarray, target_ratio: np.float, *args, **kwargs) -> np.ndarray:
    return


def windowbox(img: np.ndarray, target_ratio: np.float, *args, **kwargs) -> np.ndarray:
    return


def rotate(img: np.ndarray, angle: np.float, *args, **kwargs) -> np.ndarray:
    return


def crop(
    img: np.ndarray,
    top_left: tp.Tuple[int, int],
    bottom_right: tp.Tuple[int, int],
    *args,
    **kwargs
) -> np.ndarray:
    return


def scale(img: np.ndarray, *args, **kwargs):
    return
