import math
import pprint
import typing as tp
import pathlib

import argparse
import numpy as np
import cv2 as cv
import yaml

from video_onmf import utils


def _fill_effect(img,
                 fill: tp.Optional[str] = 'blur'
                 ):
    if fill == "blur":
        gauss = cv.getGaussianKernel(20, 10)
        kernel = gauss * gauss.transpose(1, 0)
        effect = color_filter(img, kernel=kernel)
    elif fill == "white":
        effect = np.full(img.shape, 255, dtype=np.uint8)
    else:
        effect = np.zeros_like(img)
    return effect


def color_filter(img,
                 kernel: np.ndarray
                 ) -> np.ndarray:

    filtered = (cv.filter2D(src=img,
                            ddepth=-1,
                            kernel=kernel
                            ),)
    return filtered[0]


class NeatImage:
    def __init__(self, im: np.ndarray):
        self.im = im

    @classmethod
    def read(cls, filepath: tp.Union[str, pathlib.Path]):
        return cls(utils.ird(pathlib.Path(filepath)))

    def show(self):
        return utils.ish(self.im)

    def save(self, filepath: tp.Union[str, pathlib.Path]):
        return utils.isv(self.im, pathlib.Path(filepath))

    def flip_horizontal(self):
        self.im = cv.flip(self.im, 1)
        return self

    def flip_vertical(self):
        self.im = cv.flip(self.im, 0)
        return self

    def pillarbox(
        self,
        target_ratio: float,
        fill: str,
    ):

        height, width, *_ = self.im.shape
        target_width = int(target_ratio * height)
        source_ratio = width / height

        if target_ratio <= source_ratio:
            return self

        background = _fill_effect(self.im, fill)
        background = NeatImage(background).scale_to_fit(height, target_width)

        skip_cols_left = (target_width - width) // 2
        skip_cols_right = target_width - width - skip_cols_left

        col_slice = slice(skip_cols_left, target_width - skip_cols_right)

        background.im[:, col_slice, :] = self.im[:, :, :]

        self.im = np.copy(background.im)
        return self

    def letterbox(
        self,
        target_ratio: tp.Optional[float] = None,
        fill: tp.Optional[str] = None,
    ):

        height, width, *_ = self.im.shape

        target_height = int(target_ratio * width)
        source_ratio = width / height
        if target_ratio == source_ratio:
            return self

        background = _fill_effect(self.im, fill)
        background = NeatImage(background).scale_to_fit(target_height, width)

        skip_rows_top = (target_height - height) // 2
        skip_rows_bottom = target_height - height - skip_rows_top

        row_slice = slice(skip_rows_top, target_height - skip_rows_bottom)

        background.im[row_slice, :, :] = self.im[:, :, :]

        self.im = np.copy(background.im)
        return self

    def rotate(
        self,
        angle: tp.Optional[float] = None,
    ):

        rotation_angle = angle * 180 / math.pi
        height, width, *_ = self.im.shape
        image_center = int(height / 2), int(width / 2)
        rot_mat = cv.getRotationMatrix2D(image_center, rotation_angle, 1.0)
        result = cv.warpAffine(
            self.im, rot_mat, self.im.shape[1::-1], flags=cv.INTER_LINEAR
        )
        self.im = np.copy(result)
        return self

    def windowbox(
        self,
        height_to_add: int,
        width_to_add: int,
        fill: str
    ):

        height, width, *_ = self.im.shape

        rows_top = height_to_add // 2
        rows_bottom = height_to_add - rows_top

        cols_left = width_to_add // 2
        cols_right = width_to_add - cols_left

        background = _fill_effect(self.im, fill)
        background = NeatImage(background).scale_to_fit(height + height_to_add, width + width_to_add)

        row_slice = slice(rows_top, height + height_to_add - rows_bottom)
        col_slice = slice(cols_left, width + width_to_add - cols_right)

        background.im[row_slice, col_slice, :] = self.im[:, :, :]

        self.im = np.copy(background.im)
        return self

    def crop(
        self,
        rows: int,
        columns: int
    ):

        height, width, *_ = self.im.shape

        top_rem = int(rows / 2)
        bottom_rem = rows - top_rem

        left_rem = int(columns / 2)
        right_rem = columns - left_rem

        if rows >= height + 20 or columns >= width + 20:
            return self.im

        row_slice = slice(top_rem, height - bottom_rem)
        col_slice = slice(left_rem, width - right_rem)
        self.im = np.copy(self.im[row_slice, col_slice, :])
        return self

    def scale_uniform(
        self,
        ratio: float,
    ):
        return self.scale(
            x=ratio,
            y=ratio,
        )

    def scale(
        self,
        x: float,
        y: float,
    ):

        height, width, *_ = self.im.shape
        result = cv.resize(self.im, (0, 0), x, y, interpolation=cv.INTER_LINEAR)
        self.im = np.copy(result)
        return self

    def scale_to_fit(
        self,
        height: int,
        width: int,
    ):

        result = cv.resize(self.im, (width, height), interpolation=cv.INTER_LINEAR)
        self.im = np.copy(result)
        return self

    @classmethod
    def process(cls,
                img: np.ndarray,
                config,
                actions
                ) -> np.ndarray:

        config_img = config['perturbation']['image']

        neat = NeatImage(img)
        for action in actions:
            func = getattr(neat, action)
            kwargs = {}
            if action in config_img:
                kwargs = config_img[action]
            neat = func(**kwargs)

        return neat.im

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=pathlib.Path,
        help="""
            Path to input image/video.
        """,
        required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="""
            Path to output image/video.
        """,
        required=True
    )

    parser.add_argument(
        "-c",
        "--config",
        type=pathlib.Path,
        help="""
            Path to the config file. If not provided,
            then default config is used. Otherwise, the
            default config is overridden based on the
            provided properties.
        """
    )

    ignore = {'show', 'read', 'save', 'process'}

    for stuff in dir(NeatImage):
        attr = getattr(NeatImage, stuff)
        if stuff.startswith("_"):
            continue
        name = getattr(attr, '__name__')
        if name in ignore:
            continue
        parser.add_argument(
            f"--{name.replace('_','-')}",
            action=argparse.BooleanOptionalAction,
            help=attr.__doc__,
            default=False
        )
    return parser


def _load_config(path: pathlib.Path):
    with open(path, 'r') as f:
        config_ = yaml.load(f.read(), Loader=yaml.Loader)
    return config_


def parse_actions(**kwargs):
    to_skip = {
        'input',
        'output',
        'config',
    }
    actions = []
    for k, v in kwargs.items():
        if k in to_skip:
            continue
        if v:
            actions.append(k)
    return actions


def main():
    parser = create_parser()
    args = parser.parse_args()

    config = _load_config(pathlib.Path("./config.default.yaml"))
    config.update(_load_config(args.config))

    img = utils.ird(args.input)

    if img is None:
        print(f"Unable to read image at {args.input}")

    actions = parse_actions(**args.__dict__)
    result = NeatImage.process(img, config, actions)

    utils.isv(result, args.output)


if __name__ == '__main__':
    main()


# class Perturbation:
#
#     def __init__(self,
#                  config_path: pathlib.Path = pathlib.Path("./config.default.yaml")
#                  ):
#         legit_default_config = _load_config(pathlib.Path("./config.default.yaml"))
#         self.config = legit_default_config["perturbation"]["image"]
#
#         _config = _load_config(config_path)
#         self.config.update(_config["perturbation"]["image"])
#
#     def flip_horizontal(self,
#                         img: np.ndarray
#                         ) -> np.ndarray:
#         return cv.flip(img, 1)
#
#     def flip_vertical(self,
#                       img: np.ndarray
#                       ) -> np.ndarray:
#         return cv.flip(img, 0)
#
#
#
#
#
#
#
#
#
#
#
#
# def _pathifier(p: Perturbation):
#     """A really ugly way to enhance all perturbations.
#
#     The perturbations are designed to work with numpy arrays,
#     not paths to the media itself. So this wrapper produces
#     a class that mimics the behavior of the Perturbation class
#     but allows some I/O for CLI invocations.
#
#     """
#     from video_onmf import utils
#
#     source_methods = {
#         k: getattr(p, k) for k in dir(p) if not k.startswith('_') and callable(getattr(p, k))
#     }
#
#     def init(self,
#              config_path: tp.Optional[pathlib.Path] = pathlib.Path("./config.default.yaml")
#              ):
#         _config = _load_config(config_path)
#         self.config = _config["perturbation"]["image"]
#
#     def modify_method(func):
#         def blah(
#                  self,
#                  img,
#                  output: tp.Optional[pathlib.Path] = None,
#                  *args,
#                  **kwargs
#                  ):
#
#             if isinstance(img, pathlib.Path) or isinstance(img, str):
#                 img = utils.ird(pathlib.Path(img))
#             else:
#                 print(f"The argument '{img}' does not represent a file.")
#                 return
#             if img is None:
#                 print(f"The media could not be read.")
#                 return
#
#             result = func(img, *args, **kwargs)
#             if output is not None:
#                 utils.isv(result, output_path=pathlib.Path(output))
#             return result
#
#         return blah
#
#     target_methods = {}
#     for k, v in source_methods.items():
#         target_methods[k] = modify_method(v)
#
#     classname = 'PerturberForFiles'
#     inherits = ()
#     initial_dict = {
#         **target_methods,
#         'config': p.config,
#         '__init__': init,
#         '__doc__': p.__doc__
#     }
#
#     return type(classname, inherits, initial_dict)
#
#
# if __name__ == "__main__":
#     perturber = Perturbation()
#     fire.Fire(_pathifier(perturber))
