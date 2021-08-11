import functools
import math
import pdb

import numpy as np
import cv2 as cv
import typing as tp
import fire
import functools
import pathlib


def clean(func):
    @functools.wraps(func)
    def check_path_or_img(*args, **kwargs):
        pdb.set_trace()
        if 'img' in kwargs:
            img = kwargs.pop('img', None)
        else:
            img = args[1]
            args = tuple(list(args)[2:])
        if isinstance(img, np.ndarray):
            return func(img, *args, **kwargs)
        elif isinstance(img, str) or isinstance(img, pathlib.Path):
            img = pathlib.Path(img)
            return func(utils.ird(img), *args, **kwargs)
        return None
    return check_path_or_img


class Perturbation:
    def __init__(self):
        pass

    @clean
    def flip_horizontal(self, img: np.ndarray, *args, **kwargs) -> np.ndarray:
        return cv.flip(img, 1)

    @clean
    def flip_vertical(self, img: np.ndarray, *args, **kwargs) -> np.ndarray:
        return cv.flip(img, 0)

    @clean
    def color_filter(self,
                     img: np.ndarray,
                     filter_kernel: np.ndarray,
                     *args,
                     **kwargs
                     ) -> np.ndarray:
        filtered = cv.filter2D(src=img, ddepth=-1, kernel=filter_kernel),
        return filtered[0]

    @clean
    def pillarbox(self,
                  img: np.ndarray,
                  pillarbox_target_ratio: float,
                  *args,
                  **kwargs
                  ) -> np.ndarray:

        height, width, *_ = img.shape
        target_width = int(pillarbox_target_ratio * height)
        source_ratio = width / height

        if pillarbox_target_ratio < source_ratio:
            return img

        gauss = cv.getGaussianKernel(20, 10)
        kernel = gauss * gauss.transpose(1, 0)

        blurred = self.color_filter(img, filter_kernel=kernel, *args, **kwargs)
        background = self.scale_to_fit(blurred, height, target_width, *args, **kwargs)

        skip_cols_left = (target_width - width) // 2
        skip_cols_right = target_width - width - skip_cols_left

        background[:, skip_cols_left: target_width - skip_cols_right, :] = img[:, :, :]

        return background

    @clean
    def letterbox(self,
                  img: np.ndarray,
                  letterbox_target_ratio: float,
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

            blurred = self.color_filter(img, filter_kernel=kernel, *args, **kwargs)
        elif fill == 'white':
            blurred = np.full(img.shape, 255, dtype=np.uint8)
        else:
            blurred = np.zeros_like(img)

        background = self.scale_to_fit(blurred, target_height, width, *args, **kwargs)

        skip_rows_top = (target_height - height) // 2
        skip_rows_bottom = target_height - height - skip_rows_top

        background[skip_rows_top: target_height - skip_rows_bottom, :, :] = img[:, :, :]

        return background

    @clean
    def windowbox(self,
                  img: np.ndarray,
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
            blurred = self.color_filter(img, filter_kernel=kernel)

        elif fill == 'white':
            blurred = np.full(img.shape, 255, dtype=np.uint8)
        else:
            blurred = np.zeros_like(img)

        background = self.scale_to_fit(blurred, height + height_to_add, width + width_to_add)

        background[rows_top: height + height_to_add - rows_bottom, cols_left: width + width_to_add - cols_right, :] = img[:, :, :]

        return background

    @clean
    def rotate(self,
               img: np.ndarray,
               rotation_angle: float,
               *args,
               **kwargs
               ) -> np.ndarray:
        rotation_angle = rotation_angle * 180 / math.pi
        height, width, *_ = img.shape
        image_center = int(height / 2), int(width / 2)
        rot_mat = cv.getRotationMatrix2D(image_center, rotation_angle, 1.0)
        result = cv.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)
        return result

    @clean
    def crop(
        self,
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

    @clean
    def scale_uniform(
        self,
        img: np.ndarray,
        scale_ratio: float,
        *args,
        **kwargs
    ) -> np.ndarray:
        return self.scale(img,
                          scale_x=scale_ratio,
                          scale_y=scale_ratio,
                          *args,
                          **kwargs
                          )

    @clean
    def scale(
        self,
        img: np.ndarray,
        scale_x: float,
        scale_y: float,
        *args,
        **kwargs
    ) -> np.ndarray:
        height, width, *_ = img.shape
        return cv.resize(img, (0, 0), fx=scale_x, fy=scale_y, interpolation=cv.INTER_LINEAR)

    @clean
    def scale_to_fit(
        self,
        img: np.ndarray,
        target_height: int,
        target_width: int,
        *args,
        **kwargs
    ) -> np.ndarray:
        return cv.resize(img, (target_width, target_height), interpolation=cv.INTER_LINEAR)


if __name__ == '__main__':
    fire.Fire(Perturbation)
