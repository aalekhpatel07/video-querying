"""Utils to toy around with images.

Some short-hand utils for image-first
development. These are wrappers of cv2
with some extra functionalities.

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from typing import List, Union


def ird(img_path: Path) -> np.ndarray:
    """Read an image at the given path.

    Args:
        img_path: The path to an image.

    Returns:
        A numpy array representing the image.
    """
    return cv2.imread(str(img_path))


def ish(*imgs: List[np.ndarray]):
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


def isv(arr: np.ndarray, output_path: Union[str, Path], **kwargs):
    """Save an image in RGB channels.

    Given an image and an output path,
    save the image at the path after converting
    it to RGB.

    Args:
        arr: The numpy array that represents an image.
        output_path: The path to the output image.
        **kwargs: Any extra keyword arguments that are passed to `plt.imsave`.
    """
    path = Path(output_path)
    color_corrected = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    plt.imsave(path, color_corrected, format=kwargs.get("format", "jpeg"), **kwargs)
    return
