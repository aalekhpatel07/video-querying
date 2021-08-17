import enum
import math
import collections
import mimetypes
import functools
import itertools
import random
import typing as tp
import pathlib
import argparse

import skimage.measure
import numpy as np
import cv2 as cv
import yaml
from core.video_onmf import utils
from core.video_onmf import frames as fm


def generators_factory(iterable):
    """Make generators clonable.
    https://stackoverflow.com/a/55762535/7872809

    """
    it = iter(iterable)
    deques = []
    already_gone = []

    def new_generator():
        new_deque = collections.deque()
        new_deque.extend(already_gone)
        deques.append(new_deque)

        def gen(mydeque):
            while True:
                if not mydeque:  # when the local deque is empty
                    try:
                        newval = next(it)  # fetch a new value and
                        already_gone.append(newval)
                        for d in deques:  # load it to all the deques
                            d.append(newval)
                    except StopIteration as serr:
                        break
                yield mydeque.popleft()

        return gen(new_deque)

    return new_generator


def _fill_effect(img, fill: tp.Optional[str] = "blur"):
    if fill == "blur":
        gauss = cv.getGaussianKernel(20, 10)
        kernel = gauss * gauss.transpose(1, 0)
        effect = color_filter(img, kernel=kernel)
    elif fill == "white":
        effect = np.full(img.shape, 255, dtype=np.uint8)
    else:
        effect = np.zeros_like(img)
    return effect


def color_filter(img, kernel: np.ndarray) -> np.ndarray:
    return cv.filter2D(src=img, ddepth=-1, kernel=kernel)


class Kernel:
    SEPIA = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])


class NeatImage:
    def __init__(self, im: np.ndarray):
        self.im = im

    @classmethod
    def read(cls, filepath: tp.Union[str, pathlib.Path]):
        return cls(utils.ird(pathlib.Path(filepath)))

    def show(self):
        return utils.ish(self.im)

    def save(self, filepath: tp.Union[str, pathlib.Path]):
        if self.im is None:
            print(f"Could not save image at {filepath} because it is None.")
            return
        return utils.isv(self.im, pathlib.Path(filepath))

    def blur(self, ksize: int, sigma: float):
        gauss = cv.getGaussianKernel(ksize, sigma)
        kernel = gauss * gauss.transpose(1, 0)
        effect = color_filter(self.im, kernel=kernel)
        im = np.copy(effect)
        return NeatImage(im)

    def flip_horizontal(self):
        im = cv.flip(self.im, 1)
        return NeatImage(im)

    def flip_vertical(self):
        im = cv.flip(self.im, 0)
        return NeatImage(im)

    def sepia(self):
        im = color_filter(self.im, kernel=Kernel.SEPIA)
        return NeatImage(im)

    def grayscale(self):
        im = cv.cvtColor(self.im, cv.COLOR_BGR2GRAY)
        return NeatImage(im)

    def embed_descriptors(self, top: int):
        gray = cv.cvtColor(self.im, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp = sift.detect(gray, None)
        kp = sorted(kp, key=lambda x: x.response, reverse=True)[:top]
        im = cv.drawKeypoints(gray, kp, self.im, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return NeatImage(im)

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

        im = np.copy(background.im)
        return NeatImage(im)

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

        im = np.copy(background.im)
        return NeatImage(im)

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
        im = np.copy(result)
        return NeatImage(im)

    def windowbox(self, height_to_add: int, width_to_add: int, fill: str):

        height, width, *_ = self.im.shape

        rows_top = height_to_add // 2
        rows_bottom = height_to_add - rows_top

        cols_left = width_to_add // 2
        cols_right = width_to_add - cols_left

        background = _fill_effect(self.im, fill)
        background = NeatImage(background).scale_to_fit(
            height + height_to_add, width + width_to_add
        )

        row_slice = slice(rows_top, height + height_to_add - rows_bottom)
        col_slice = slice(cols_left, width + width_to_add - cols_right)

        background.im[row_slice, col_slice, :] = self.im[:, :, :]

        im = np.copy(background.im)
        return NeatImage(im)

    def crop(self, rows: int, columns: int):

        height, width, *_ = self.im.shape

        top_rem = int(rows / 2)
        bottom_rem = rows - top_rem

        left_rem = int(columns / 2)
        right_rem = columns - left_rem

        if rows >= height + 20 or columns >= width + 20:
            return self

        row_slice = slice(top_rem, height - bottom_rem)
        col_slice = slice(left_rem, width - right_rem)
        im = np.copy(self.im[row_slice, col_slice, :])
        return NeatImage(im)

    def scale_uniform(
            self,
            ratio: float,
    ):
        cp = NeatImage(self.im)
        return cp.scale(
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
        im = np.copy(result)
        return NeatImage(im)

    def scale_to_fit(
            self,
            height: int,
            width: int,
    ):

        result = cv.resize(self.im, (width, height), interpolation=cv.INTER_LINEAR)
        im = np.copy(result)
        return NeatImage(im)

    @classmethod
    def process(cls, filepath: pathlib.Path, config, actions) -> "NeatImage":

        neat = NeatImage.read(filepath)
        if neat is None or neat.im is None:
            return NeatImage(np.array([]))
        for action in actions:
            func = getattr(neat, action)
            kwargs = {}
            if action in config:
                kwargs = config[action]
            neat = func(**kwargs)

        return neat


def _extend(func):
    @functools.wraps(func)
    def blah(self, *args, **kwargs):
        iterable = (func(frame, *args, **kwargs).im for frame in self.factory())
        return type(self)(iterable, metadata=self.metadata)

    return blah


class NeatVideo:
    extended = {
        "pillarbox",
        "blur",
        "letterbox",
        "windowbox",
        "scale",
        "scale_to_fit",
        "scale_uniform",
        "crop",
        "rotate",
        "flip_horizontal",
        "flip_vertical",
        "embed_descriptors",
        "sepia",
        "grayscale"
    }

    def __init__(
            self,
            frame_stream: tp.Iterable[np.ndarray],
            metadata: tp.Optional[tp.Dict] = None,
    ):
        self.factory = generators_factory(map(NeatImage, frame_stream))
        self.metadata = metadata

        for action in self.extended:
            setattr(self, action, _extend(getattr(NeatImage, action)))

    def to_ndarray_generator(self) -> tp.Iterable[np.ndarray]:
        return map(lambda x: x.im, self.factory())

    @classmethod
    def read(cls, filepath: tp.Union[str, pathlib.Path], **kwargs):
        metadata = fm.probe(cv.VideoCapture(str(filepath)))
        return cls(fm.from_video(filepath), metadata)

    def show(self, *args, **kwargs):
        return utils.vsh(self.factory(), *args, **kwargs)

    def save(self, filepath: tp.Union[str, pathlib.Path]):
        fps = self.metadata["fps"]

        return utils.vsv(
            map(lambda x: x.im, self.factory()),
            fps,
            filepath,
        )

    def screenshot(
            self, frame: int
    ) -> tp.Optional[NeatImage]:

        shot = next(itertools.islice(self.factory(), frame, None), None)
        if shot is not None:
            return NeatImage(im=shot.im)
        return None

    def keyframe_indices(self):
        entropies = [
            skimage.measure.shannon_entropy(img.im)
            for img in self.factory()
        ]

        peaks = np.r_[True, entropies[1:] < entropies[:-1]] & np.r_[entropies[:-1] < entropies[1:], True]
        return np.where(peaks)[0]

    def screenshot_many(
        self, frames: tp.Iterable[int]
    ) -> tp.Optional[tp.Iterable[NeatImage]]:

        check = set(frame for frame in frames)
        for idx, frame in enumerate(self.factory()):
            if idx in check:
                yield NeatImage(frame)

    def adjust_speed(self, scale: float):

        metadata = dict(**self.metadata)
        metadata.update({"fps": self.metadata["fps"] * scale})

        return NeatVideo(
            map(lambda x: x.im, self.factory()),
            metadata=metadata
        )

    def adjust_duration(self, scale: float):
        return self.adjust_speed(1. / scale)

    def adjust_frames(self, fps):
        if fps > self.metadata["fps"]:
            return self

        duration = self.metadata["total_frames"] / self.metadata["fps"]

        frames_to_keep = int(duration * fps)
        indices = np.round(
            np.linspace(0, self.metadata["total_frames"] - 1, frames_to_keep)
        ).astype(int)
        indices = set(indices)

        updated_stream = (
            neat_image
            for (idx, neat_image) in enumerate(self.factory())
            if idx in indices
        )

        metadata = dict(**self.metadata)
        metadata.update({
            "total_frames": len(indices),
            "fps": fps
        })

        return NeatVideo(map(lambda x: x.im, updated_stream),
                         metadata=metadata
                         )

    def rearrange_group(self, group_size: int):

        gops = fm.from_stream_grouped(self.factory(), group_size)
        gops = [list(x) for x in gops]
        random.shuffle(gops)

        updated = (NeatImage(frame) for frame in itertools.chain.from_iterable(gops))
        return NeatVideo(map(lambda x: x.im, updated), self.metadata)

    def crop_duration(self, start: tp.Dict[str, tp.Any], end: tp.Dict[str, tp.Any]):
        s_hour = int(start["hour"] or 0)
        s_minute = int(start["minute"] or 0)
        s_second = float(start["second"] or 0)

        f_hour = int(end["hour"] or 0)
        f_minute = int(end["minute"] or 0)
        f_second = float(end["second"] or 0)

        total_seconds_to_start = (60 ** 2) * s_hour * 60 * s_minute + s_second
        total_seconds_to_end = (60 ** 2) * f_hour * 60 * f_minute + f_second

        start_frame = int(self.metadata["fps"] * total_seconds_to_start)
        start_frame = max(0, start_frame)

        end_frame = int(self.metadata["fps"] * total_seconds_to_end)
        end_frame = min(self.metadata["total_frames"], end_frame)

        updated_stream = (
            neat_image
            for (idx, neat_image) in enumerate(self.factory())
            if start_frame <= idx <= end_frame
        )

        metadata = dict(**self.metadata)
        metadata.update({
            "total_frames": end_frame - start_frame + 1,
        })

        return NeatVideo(
            map(lambda x: x.im, updated_stream),
            metadata=metadata
        )

    @classmethod
    def process(
            cls, filepath: pathlib.Path, config, actions, **kwargs
    ) -> tp.Union[NeatImage, "NeatVideo"]:
        neat = NeatVideo.read(filepath=filepath, **kwargs)
        for action in actions:
            func = getattr(neat, action)
            kwargs = {}
            if action in config:
                kwargs = config[action]
            if action in cls.extended:
                neat = func(neat, **kwargs)
            else:
                neat = func(**kwargs)
        return neat


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=pathlib.Path,
        help="""
            Path to input image/video.
        """,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="""
            Path to output image/video.
        """,
        required=True,
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
        """,
    )

    ignore = {"show", "read", "save", "process", "to_ndarray_generator", "extended"}

    neat_image_dir = set(dir(NeatImage))
    processable_video_dir = set(dir(NeatVideo))

    chained = itertools.chain(
        neat_image_dir, processable_video_dir - NeatVideo.extended
    )

    def filter_fn(x):
        return not x.startswith("_") and x.replace("-", "_") not in ignore

    for stuff in filter(filter_fn, chained):
        try:
            attr = getattr(NeatImage, stuff)
        except AttributeError:
            attr = getattr(NeatVideo, stuff)
        name = getattr(attr, "__name__")
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            action=argparse.BooleanOptionalAction,
            help=attr.__doc__,
            default=False,
        )
    return parser


def _load_config(path: pathlib.Path):
    try:
        with open(path, "r") as f:
            config_ = yaml.load(f.read(), Loader=yaml.Loader)
    except TypeError:
        return {}
    return config_


def parse_actions(**kwargs):
    to_skip = {
        "input",
        "output",
        "config",
    }
    actions = []
    for k, v in kwargs.items():
        if k in to_skip:
            continue
        if v:
            actions.append(k)
    return actions


def _default_config():
    with open("./perturb_config.default.yaml", "r") as f:
        big_str = f.read()
    return yaml.load(big_str, Loader=yaml.Loader)


class MimeType(enum.Enum):
    VIDEO = enum.auto()
    IMAGE = enum.auto()
    VIDEO_YUV = enum.auto()


def media_check(path: pathlib.Path) -> tp.Optional[MimeType]:
    if path.suffix == ".yuv":
        return MimeType.VIDEO_YUV
    mime = mimetypes.guess_type(path)[0]
    if mime is None:
        return None
    if mime.startswith("video"):
        return MimeType.VIDEO
    elif mime.startswith("image"):
        return MimeType.IMAGE
    return None


def main():
    parser = create_parser()
    args = parser.parse_args()
    config = _default_config()

    config_img = config["image"] or {}
    config_vid = config["video"] or {}

    config = dict(**config_img, **config_vid)
    config_passed = _load_config(args.config)

    config.update(**(config_passed["image"] if "image" in config_passed else {}))
    config.update(**(config_passed["video"] if "video" in config_passed else {}))

    actions = parse_actions(**args.__dict__)

    given_mime = media_check(args.input)

    if given_mime is None:
        print(f"Unable to read media at {args.input}")
        return

    if given_mime is MimeType.IMAGE:
        result = NeatImage.process(args.input, config, actions)
    elif given_mime is MimeType.VIDEO:
        result = NeatVideo.process(args.input, config, actions)
    else:
        result = NeatVideo.process(args.input, config, actions, size=(352, 288))

    result.save(args.output)


if __name__ == "__main__":
    main()
