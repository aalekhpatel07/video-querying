import time
import pathlib
import os
import mimetypes
import argparse
import typing as tp
import inspect
import functools
import math
import random
from copy import deepcopy
from pprint import pprint as pp
import numpy as np
import uuid
import msgpack as mp
import tqdm

import cv2 as cv

from core.video_onmf import frames as fm
from core.video_onmf import utils
from core.perturb import image
# from core.perturb import video


class Action:
    def __init__(self, name: str, func: tp.Callable, *args, **kwargs):
        self.name = name
        self.args = args
        self.func = func
        self.kwargs = kwargs

    def to_dict(self):
        return {
            'name': self.name,
            'args': self.args,
            'kwargs': self.kwargs,
            'func': self.func
        }

    def __str__(self):
        return str(self.to_dict())

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class Config:
    # config = {
    #     'filter_kernel': np.ndarray([1], dtype=np.uint8),
    #     'pillarbox_target_ratio': 4/3,
    #     'letterbox_target_ratio': 16/9,
    #     'windowbox_target_ratio': 4/3,
    #     'rotation_angle': math.pi / 2,
    #     'top_left_crop_coord': (0, 0),
    #     'bottom_right_crop_coord': (None, None)
    # }


    def __init__(self,
                 pillarbox_target_ratio: float = 4/3,
                 letterbox_target_ratio: float = 16/9,
                 windowbox_target_ratio: float = 4/3,
                 rotation_angle: float = math.pi / 2,
                 remove_rows: int = 0,
                 remove_cols: int = 0,
                 scale_ratio: float = 1,
                 scale_x: float = 1,
                 scale_y: float = 1,
                 target_height: int = 1000,
                 target_width: int = 1000,
                 filter_kernel: np.ndarray = np.array([1]),
                 fill: str = "blur",
                 **kwargs
                 ):
        self.filter_kernel = filter_kernel
        self.pillarbox_target_ratio = pillarbox_target_ratio
        self.letterbox_target_ratio = letterbox_target_ratio
        self.windowbox_target_ratio = windowbox_target_ratio
        self.rotation_angle = rotation_angle
        self.remove_rows = remove_rows
        self.remove_cols = remove_cols
        self.scale_ratio = scale_ratio
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.target_height = target_height
        self.target_width = target_width
        self.fill = fill

        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__dict__)

    def update(self, config):
        self.__dict__.update(config)

    @classmethod
    def _generate_attr(cls, attribute):
        if attribute.endswith('_target_ratio'):
            return random.random() * 2
        if attribute.endswith('_angle'):
            return random.random() * 2 * math.pi
        if attribute.endswith('filter_kernel'):
            gauss = cv.getGaussianKernel(random.randint(1, 100), random.randint(1, 20))
            kernel = gauss * gauss.transpose(1, 0)
            return kernel
        if attribute.startswith('remove_'):
            return random.randint(0, 50)
        if attribute.startswith('scale_'):
            return random.random() * 2
        if attribute == "target_width" or attribute == "target_height":
            return random.randint(0, 1000)
        if attribute == "fill":
            options = ["white", "dark", "blur"]
            return random.choice(options)
        return None

    def get_random(self, **kwargs):
        fixed_values = { k: v for k, v in kwargs.items()}

        all_attributes = self.to_dict().keys()

        updated = {}
        to_change = all_attributes - set(fixed_values.keys())
        for k in to_change:
            updated[k] = self._generate_attr(k)
        return Config(**fixed_values, **updated)

    def to_dict(self):
        all_attributes = {}
        for k in self.__dict__:
            if k.startswith('_') or callable(getattr(self, k)):
                continue
            if inspect.ismethod(getattr(self, k)):
                continue
            all_attributes[k] = getattr(self, k)
        return all_attributes


def get_actions(mime_type: str) -> tp.Dict[str, tp.Callable]:
    actions = {}
    if mime_type == "image":
        mod = image
    else:
        mod = video

    for attr in dir(mod):
        if (
            not attr.startswith("_")
            and hasattr(mod, attr)
            and callable(getattr(mod, attr))
        ):
            actions[attr] = getattr(mod, attr)
    return actions


all_actions = get_actions("image")


def chain_actions(actions: tp.Iterable[Action], *args, **kwargs):
    def chained(img: np.ndarray) -> np.ndarray:
        current = img
        for action in actions:
            current = action(current, *args, **kwargs)
        return current
    return chained


def perturb_video(
    frame_stream: tp.Generator[np.ndarray, None, None],
    actions: tp.Optional[tp.List[tp.Callable]] = None,
    *args,
    **kwargs
):

    if actions is None:
        actions = random.choices(
            list(all_actions.items()),
            k=random.randint(1, 4)
        )
    clean_actions = [
        Action(act, *args, func=func, **kwargs) for (act, func) in actions
    ]
    chained = chain_actions(clean_actions, *args, **kwargs)
    return (chained(frame) for frame in frame_stream), clean_actions


def perturb_image(
    img: np.ndarray,
    actions: tp.Optional[tp.List[tp.Callable]] = None,
    *args,
    **kwargs
):
    if actions is None:
        actions = random.choices(
            list(all_actions.items()),
            k=random.randint(1, 6)
        )

    clean_actions = [
        Action(act, *args, func=func, **kwargs) for (act, func) in actions
    ]
    chained = chain_actions(clean_actions, *args, **kwargs)
    return chained(img), clean_actions


def read_parser():
    parser = argparse.ArgumentParser(description="Create a dataset of media.")
    parser.add_argument(
        "-s",
        "--source",
        type=pathlib.Path,
        help="The source directory that contains all the media.",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=pathlib.Path,
        help="The directory for storing the modified media in.",
        required=True,
    )

    return parser.parse_args()


def media_files(
    directory: pathlib.Path,
) -> tp.Generator[tp.Tuple[str, pathlib.Path], None, None]:
    for root, dirs, files in os.walk(directory):
        for file in files:
            mime_type = mimetypes.guess_type(os.path.join(root, file))[0]
            if mime_type.startswith("video"):
                yield "video", pathlib.Path(os.path.join(root, file))
            elif mime_type.startswith("image"):
                yield "image", pathlib.Path(os.path.join(root, file))
        for d in dirs:
            yield from media_files(pathlib.Path(os.path.join(root, d)))


def read_report(path: pathlib.Path):
    with open(path, 'rb') as f:
        return mp.unpack(f)


def main():
    res = read_report(pathlib.Path("./data/output/report.mp"))
    print(res["timestamp"])
    for x in res["files"]:
        print(x["filename"])
        for act in x["actions"]:
            print(act)
        print("--------")
    return res

def do_your_thing():
    args = read_parser()
    source, dest = args.source, args.output_dir

    number_of_alterations_per_image: int = 5
    number_of_alterations_per_video: int = 1
    c = Config()

    outer = {
        "source": str(source),
        "destination": str(dest),
        "timestamp": time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    }
    report = []

    for mime, path in tqdm.tqdm(media_files(source), desc="Source"):
        # transformed = None
        # updated = None
        report_item = {
            "parent": path.stem + path.suffix,
        }

        if mime == "image":
            img = utils.ird(path)

            for _ in range(number_of_alterations_per_image):
                current_path = (dest / str(uuid.uuid4())).with_suffix(path.suffix)

                config = c.get_random()
                updated, actions = perturb_image(img, **config.to_dict())

                report_item["filename"] = current_path.stem + current_path.suffix
                report_item["actions"] = list(map(str, actions))

                utils.isv(updated, current_path)
                report.append(deepcopy(report_item))

        elif mime == "video":
            cap = cv.VideoCapture(str(path))
            metadata = fm.probe(cap)
            for _ in tqdm.tqdm(range(number_of_alterations_per_video), desc="perturbation"):
                current_path = (dest / str(uuid.uuid4())).with_suffix(path.suffix)

                config = c.get_random(fill="blur", height_to_add=100, width_to_add=200)
                vid = fm.from_video(path)
                updated, actions = perturb_video(
                    vid,
                    actions=[("windowbox", all_actions["windowbox"])],
                    **config.to_dict()
                )

                report_item["filename"] = current_path.stem + current_path.suffix
                report_item["actions"] = list(map(str, actions))

                utils.vsv(updated,
                          path=current_path,
                          fps=metadata["fps"],
                          )
                report.append(deepcopy(report_item))

    outer["files"] = report

    with open(dest / "report.mp", 'wb') as f:
        mp.pack(outer, f)


if __name__ == "__main__":
    main()
    # do_your_thing()
