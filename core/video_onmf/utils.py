"""Utils to toy around with images.

Some short-hand utils for image-first
development. These are wrappers of cv2
with some extra functionalities.

"""
import time
import pathlib
import typing as tp
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tqdm


def ird(img_path: pathlib.Path) -> np.ndarray:
    """Read an image at the given path.

    Args:
        img_path: The path to an image.

    Returns:
        A numpy array representing the image.
    """
    return cv2.imread(str(img_path))


def ish(*imgs: tp.Union[np.ndarray, tp.Iterable[np.ndarray]]):
    """Show images.

    Show any number of images
    passed in the argument.

    Args:
        *imgs: A collection of numpy arrays
            that represent images.

    """
    for img in imgs:
        if img is not None:
            cvimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots()
            ax.imshow(cvimg)
            ax.grid(False)
            plt.show()
    pass


def isv(arr: np.ndarray, output_path: tp.Union[str, pathlib.Path], **kwargs):
    """Save an image in RGB channels.

    Given an image and an output path,
    save the image at the path after converting
    it to RGB.

    Args:
        arr: The numpy array that represents an image.
        output_path: The path to the output image.
        **kwargs: Any extra keyword arguments that are passed to `plt.imsave`.
    """
    path = pathlib.Path(output_path)
    if arr is None:
        return
    color_corrected = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    plt.imsave(path, color_corrected, format=kwargs.get("format", "jpeg"), **kwargs)
    return


def vsh(
    frame_stream: tp.Iterable[np.ndarray],
    transform_hook: tp.Optional[tp.Callable[[np.ndarray], np.ndarray]] = None,
    description: tp.Optional[str] = "1",
    width: tp.Optional[int] = None,
    height: tp.Optional[int] = None,
    interpolation: tp.Optional[tp.Any] = cv2.INTER_AREA,
):
    if transform_hook is None:
        transform_hook = lambda x: x

    for frame in transform_hook(frame_stream):
        if width is not None and height is not None:
            frame = cv2.resize(frame, (width, height), interpolation=interpolation)
        cv2.imshow(description, frame)
        time.sleep(0.01)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    return


def vsv(
    frame_stream: tp.Iterable[np.ndarray],
    fps: int,
    path: pathlib.Path,
    width: tp.Optional[int] = None,
    height: tp.Optional[int] = None,
):
    stream_it = iter(frame_stream)

    try:
        first = next(stream_it)
    except StopIteration as err:
        print(err)
        return

    if height is None or width is None:
        height, width, *_ = first.shape

    writer = cv2.VideoWriter(str(path),
                             cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                             fps,
                             (width, height)
                             )
    writer.write(first)

    for frame in tqdm.tqdm(stream_it, desc="Saving frames"):
        writer.write(frame)

    # del temp
    # while True:
    #     try:
    #         current = next(stream_it)
    #         writer.write(current)
    #     except StopIteration:
    #         break

    return
