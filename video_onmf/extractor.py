import typing as tp
import pathlib

import numpy as np
import cv2 as cv
import msgpack as mp

from . import matrix as mx
from . import descriptor as dsc


def compute_raw_descriptors_given_image(
    img: np.ndarray,
    descriptor
) -> tp.Optional[np.ndarray]:
    """Compute the given descriptors in a given image.

    Args:
        img: A numpy array representation of an image.
        descriptor: The descriptor type to extract.

    Returns:
        The extracted descriptors.
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, des = descriptor.detectAndCompute(gray, None)
    if des is None or not len(des):
        return None
    des /= np.linalg.norm(des, axis=0)
    return des


def compute_raw_descriptors(frames, descriptor) -> np.ndarray:
    """Compute and stack the descriptors from all the frames within the group.

    Args:
        descriptor: An instance of cv2.xxxx_create() that will detect and compute
        the descriptors on the frames.

    Returns:
        A numpy array that contains the descriptors stacked together.
    """

    descriptors = []
    for frame in frames:
        computed = compute_raw_descriptors_given_image(frame, descriptor)
        if computed is None:
            continue
        descriptors.append(computed.T)

    stacked = np.column_stack(descriptors)
    return stacked


class CompactDescriptorExtractor:
    def __init__(
        self,
        descriptor,
        factorizer: tp.Optional[mx.MatrixFactorizer] = None,
    ):
        self.descriptor = descriptor
        self.factorizer = factorizer
        if factorizer is None:
            self.factorizer = mx.OrthogonalNonnegativeMatrixFactorizer(
                rank=10,
                rho=.01,
                maxiter=100
            )

    def extract(
        self,
        frames: tp.Generator[np.ndarray, None, None],
        source_id: tp.Optional[str] = None
    ) -> tp.Generator[dsc.CompactDescriptorVector, None, None]:
        print(f"Extracting descriptors...")

        matrix = compute_raw_descriptors(frames, self.descriptor)
        left, _ = self.factorizer.factor(matrix)

        for cols in left.T:
            yield dsc.CompactDescriptorVector(vector=cols, source_id=source_id)

    def extract_from_video(
        self,
        grouped_frames: tp.Generator[tp.Generator[np.ndarray, None, None], None, None],
        source_id: tp.Optional[str] = None
    ) -> tp.Generator[dsc.CompactDescriptorVector, None, None]:
        for gop in grouped_frames:
            yield from self.extract(gop, source_id)

    def save(
        self,
        frames: tp.Generator[np.ndarray, None, None],
        path: str,
        source_id: tp.Optional[str] = None
    ) -> None:
        vectors = [
            vector.encode() for vector in self.extract(frames, source_id)
        ]
        with open(path, "wb") as f:
            mp.pack({
                'source_id': source_id,
                'descriptors': vectors
            }, f)

    def save_from_video(
        self,
        frames: tp.Generator[tp.Generator[np.ndarray, None, None], None, None],
        path: tp.Union[pathlib.Path, str],
        source_id: tp.Optional[str] = None
    ) -> None:
        print(f"Extracting descriptors to {path}...")
        vectors = [
            vector.encode() for vector in self.extract_from_video(frames, source_id)
        ]
        with open(path, "wb") as f:
            mp.pack({
                'source_id': source_id,
                'descriptors': vectors
            }, f)

    def load(
        self,
        path: tp.Union[pathlib.Path, str],
    ) -> tp.Tuple[str, tp.Generator[dsc.CompactDescriptorVector, None, None]]:
        with open(path, 'rb') as f:
            obj = mp.unpack(f)
            descriptors = (
                dsc.CompactDescriptorVector.decode(x) for x in obj['descriptors']
            )
        return obj['source_id'], descriptors


    def __str__(self) -> str:
        return f"<CompactDescriptorBuilder factorizer={self._factorizer}>"
