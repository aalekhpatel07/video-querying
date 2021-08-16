import itertools
import math
from operator import itemgetter
import typing as tp
import pathlib

import numpy as np
import cv2 as cv


class VideoUnReadable(Exception):
    """Indicates that a video file could not be read by cv2."""

    pass


def probe(video: cv.VideoCapture) -> tp.Dict[str, tp.Union[int, float, bool]]:
    """

    Args:
        video:

    Returns:

    """
    if not video.isOpened():
        raise VideoUnReadable()

    metadata: tp.Dict[str, tp.Union[int, float, bool]] = {
        "width": int(video.get(cv.CAP_PROP_FRAME_WIDTH)),
        "height": int(video.get(cv.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(video.get(cv.CAP_PROP_FPS)),
        "total_frames": int(video.get(cv.CAP_PROP_FRAME_COUNT)),
        "convert_to_rgb": bool(video.get(cv.CAP_PROP_CONVERT_RGB)),
    }

    return metadata


def from_video(
    video_path: tp.Union[pathlib.Path, str], **kwargs
) -> tp.Generator[np.ndarray, None, None]:
    """Generate a stream of frames from video.

    Args:
        video_path:
        **kwargs:

    Returns:

    """
    cap = cv.VideoCapture(str(video_path))
    metadata = probe(cap)

    start = kwargs.get("start", 0)
    end = kwargs.get("end", metadata["total_frames"] + 1)
    step = kwargs.get("step", 1)

    indices_to_select = (x for x in range(start, end, step))

    counter = 0
    to_select = next(indices_to_select)

    if cap.isOpened():
        while counter < end:
            is_good, frame = cap.read()
            if is_good and counter == to_select:
                yield frame
                try:
                    to_select = next(indices_to_select)
                except StopIteration:
                    break
            counter += 1

def from_video_yuv(
    video_path: tp.Union[pathlib.Path, str],
    size: tp.Tuple[int, int],
    **kwargs
):
    yield from VideoCaptureYUV(video_path, size).read()


class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width, *_ = size
        self.frame_len = int(self.width * self.height * 3 / 2)
        self.shape = (int(self.height*1.5), self.width)
        self.filename = filename

    def read(self):
        with open(self.filename, 'rb') as f:
            while True:
                try:
                    raw = f.read(self.frame_len)
                    yuv = np.frombuffer(raw, dtype=np.uint8)
                    if yuv is None or not yuv.shape or not len(yuv):
                        break
                    yuv = yuv.reshape(self.shape)
                    yield cv.cvtColor(yuv, cv.COLOR_YUV2BGR_NV21)
                except EOFError as eof:
                    break


def from_image(image_path: tp.Union[pathlib.Path, str]) -> tp.Optional[np.ndarray]:
    return cv.imread(str(image_path))


def _grouped_generator(
    stream: tp.Iterable[tp.Any], group_size: int
) -> tp.Iterable[tp.Iterable[np.ndarray]]:
    yield from itertools.groupby(
        stream, key=lambda x, c=itertools.count(): math.floor(next(c) / group_size)
    )


def from_video_grouped(
    video_path: tp.Union[pathlib.Path, str], group_size: int
) -> tp.Iterable[tp.Iterable[np.ndarray]]:
    yield from map(
        itemgetter(1), _grouped_generator(from_video(video_path), group_size)
    )


def from_stream_grouped(
    stream: tp.Iterable[np.ndarray], group_size: int
) -> tp.Iterable[tp.Iterable[np.ndarray]]:
    yield from map(itemgetter(1), _grouped_generator(stream, group_size))
