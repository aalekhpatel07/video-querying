import math
import typing as tp
import pathlib

import fire
import numpy as np
import cv2 as cv
import yaml


def _load_config(path: pathlib.Path):
    with open(path, 'r') as f:
        config_ = yaml.load(f.read(), Loader=yaml.Loader)
    return config_


class Perturbation:

    def __init__(self,
                 config_path: pathlib.Path = pathlib.Path("./config.default.yaml")
                 ):
        _config = _load_config(config_path)
        self.config = _config["perturbation"]["image"]

    def flip_horizontal(self,
                        img: np.ndarray
                        ) -> np.ndarray:
        return cv.flip(img, 1)

    def flip_vertical(self,
                      img: np.ndarray
                      ) -> np.ndarray:
        return cv.flip(img, 0)

    def color_filter(self,
                     img: np.ndarray,
                     kernel: tp.Optional[np.ndarray] = None
                     ) -> np.ndarray:

        if kernel is None:
            kernel = np.array(self.config['color_filter']['kernel'])

        filtered = (cv.filter2D(src=img, ddepth=-1, kernel=kernel),)
        return filtered[0]

    def _fill_effect(self,
                     img: np.ndarray,
                     fill: tp.Optional[str] = 'blur'
                     ):
        if fill == "blur":
            gauss = cv.getGaussianKernel(20, 10)
            kernel = gauss * gauss.transpose(1, 0)
            effect = self.color_filter(img, kernel=kernel)
        elif fill == "white":
            effect = np.full(img.shape, 255, dtype=np.uint8)
        else:
            effect = np.zeros_like(img)
        return effect

    def pillarbox(
        self,
        img: np.ndarray,
        target_ratio: tp.Optional[float] = None,
        fill: tp.Optional[str] = None,
    ) -> np.ndarray:

        if target_ratio is None:
            target_ratio = float(self.config['pillarbox']['target_ratio'])
        if fill is None:
            fill = self.config["pillarbox"]["fill"]
        height, width, *_ = img.shape
        target_width = int(target_ratio * height)
        source_ratio = width / height

        if target_ratio < source_ratio:
            return img

        background = self.scale_to_fit(
            self._fill_effect(img, fill), height, target_width
        )

        skip_cols_left = (target_width - width) // 2
        skip_cols_right = target_width - width - skip_cols_left

        col_slice = slice(skip_cols_left, target_width - skip_cols_right)

        background[:, col_slice, :] = img[:, :, :]

        return background

    def letterbox(
        self,
        img: np.ndarray,
        target_ratio: tp.Optional[float] = None,
        fill: tp.Optional[str] = None,
    ) -> np.ndarray:
        if target_ratio is None:
            target_ratio = float(self.config['letterbox']['target_ratio'])
        if fill is None:
            fill = self.config["letterbox"]["fill"]

        height, width, *_ = img.shape

        target_height = int(target_ratio * width)
        source_ratio = width / height
        if target_ratio == source_ratio:
            return img

        background = self.scale_to_fit(
            self._fill_effect(img, fill), target_height, width
        )

        skip_rows_top = (target_height - height) // 2
        skip_rows_bottom = target_height - height - skip_rows_top

        row_slice = slice(skip_rows_top, target_height - skip_rows_bottom)

        background[row_slice, :, :] = img[:, :, :]

        return background

    def windowbox(
        self,
        img: np.ndarray,
        height_to_add: tp.Optional[int] = None,
        width_to_add: tp.Optional[int] = None,
        fill: tp.Optional[str] = None,
    ) -> np.ndarray:

        if height_to_add is None:
            height_to_add = int(self.config['windowbox']['height_to_add'])
        if width_to_add is None:
            width_to_add = int(self.config['windowbox']['width_to_add'])
        if fill is None:
            fill = self.config["windowbox"]["fill"]

        height, width, *_ = img.shape

        rows_top = height_to_add // 2
        rows_bottom = height_to_add - rows_top

        cols_left = width_to_add // 2
        cols_right = width_to_add - cols_left

        background = self.scale_to_fit(
            self._fill_effect(img, fill), height +
            height_to_add, width + width_to_add
        )

        row_slice = slice(rows_top, height + height_to_add - rows_bottom)
        col_slice = slice(cols_left, width + width_to_add - cols_right)

        background[row_slice, col_slice, :] = img[:, :, :]

        return background

    def rotate(
        self,
        img: np.ndarray,
        angle: tp.Optional[float] = None,
    ) -> np.ndarray:

        if angle is None:
            angle = float(self.config['rotate']['angle'])

        rotation_angle = angle * 180 / math.pi
        height, width, *_ = img.shape
        image_center = int(height / 2), int(width / 2)
        rot_mat = cv.getRotationMatrix2D(image_center, rotation_angle, 1.0)
        result = cv.warpAffine(
            img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)
        return result

    def crop(
        self,
        img: np.ndarray,
        rows: tp.Optional[int] = None,
        columns: tp.Optional[int] = None
    ) -> np.ndarray:

        if rows is None:
            rows = int(self.config['crop']['rows'])
        if columns is None:
            columns = int(self.config['crop']['columns'])

        height, width, *_ = img.shape

        top_rem = int(rows / 2)
        bottom_rem = rows - top_rem

        left_rem = int(columns / 2)
        right_rem = columns - left_rem

        if rows >= height + 20 or columns >= width + 20:
            return img

        row_slice = slice(top_rem, height - bottom_rem)
        col_slice = slice(left_rem, width - right_rem)

        return img[row_slice, col_slice, :]

    def scale_uniform(
        self,
        img: np.ndarray,
        ratio: tp.Optional[float] = None,
    ) -> np.ndarray:

        if ratio is None:
            ratio = float(self.config['scale_uniform']['ratio'])

        return self.scale(
            img,
            x=ratio,
            y=ratio,
        )

    def scale(
        self,
        img: np.ndarray,
        x: tp.Optional[float] = None,
        y: tp.Optional[float] = None,
    ) -> np.ndarray:

        if x is None:
            x = float(self.config['scale']['x'])
        if y is None:
            y = float(self.config['scale']['y'])

        height, width, *_ = img.shape
        return cv.resize(img, (0, 0), x, y, interpolation=cv.INTER_LINEAR)

    def scale_to_fit(
        self,
        img: np.ndarray,
        height: tp.Optional[int] = None,
        width: tp.Optional[int] = None,
    ) -> np.ndarray:

        if height is None:
            height = int(self.config['scale_to_fit']['height'])
        if width is None:
            width = int(self.config['scale_to_fit']['width'])

        return cv.resize(img, (width, height), interpolation=cv.INTER_LINEAR)


def _pathifier(p: Perturbation):
    """A really ugly way to enhance all perturbations.

    The perturbations are designed to work with numpy arrays,
    not paths to the media itself. So this wrapper produces
    a class that mimics the behavior of the Perturbation class
    but allows some I/O for CLI invocations.

    """
    from video_onmf import utils

    source_methods = {
        k: getattr(p, k) for k in dir(p) if not k.startswith('_') and callable(getattr(p, k))
    }

    def init(self,
             config_path: tp.Optional[pathlib.Path] = pathlib.Path("./config.default.yaml")
             ):
        _config = _load_config(config_path)
        self.config = _config["perturbation"]["image"]

    def modify_method(func):
        def blah(
                 self,
                 img,
                 output: tp.Optional[pathlib.Path] = None,
                 *args,
                 **kwargs
                 ):

            if isinstance(img, pathlib.Path) or isinstance(img, str):
                img = utils.ird(pathlib.Path(img))
            else:
                print(f"The argument '{img}' does not represent a file.")
                return
            if img is None:
                print(f"The media could not be read.")
                return

            result = func(img, *args, **kwargs)
            if output is not None:
                utils.isv(result, output_path=pathlib.Path(output))
            return result

        return blah

    target_methods = {}
    for k, v in source_methods.items():
        target_methods[k] = modify_method(v)

    classname = 'PerturberForFiles'
    inherits = ()
    initial_dict = {
        **target_methods,
        'config': p.config,
        '__init__': init,
        '__doc__': p.__doc__
    }

    return type(classname, inherits, initial_dict)


if __name__ == "__main__":
    perturber = Perturbation()
    fire.Fire(_pathifier(perturber))
