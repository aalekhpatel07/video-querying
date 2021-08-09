import pdb
import typing as tp
import os
import mimetypes
import pathlib
import operator as op
import numpy as np
import cv2 as cv
import msgpack as mp

from . import matrix as mx
from . import descriptor as dsc
from . import frames as fm


def compute_raw_descriptors_given_image(
    img: np.ndarray, descriptor, **kwargs
) -> tp.Optional[np.ndarray]:
    """Compute the given descriptors in a given image.

    Args:
        img: A numpy array representation of an image.
        descriptor: The descriptor type to extract.

    Returns:
        The extracted descriptors.
    """
    top = kwargs.get("raw_top", 100)
    if img is None:
        return None
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kps, des = descriptor.detectAndCompute(gray, None)
    if des is None or not len(des):
        return None

    clean = sorted(zip(kps, des), key=lambda item: item[0].response, reverse=True)[:top]

    clean = np.array(list(map(op.itemgetter(1), clean)), dtype=np.float32)
    clean /= np.linalg.norm(clean, axis=0)
    return clean


def compute_raw_descriptors(frames, descriptor, **kwargs) -> tp.Optional[np.ndarray]:
    """Compute and stack the descriptors from all the frames within the group.

    Args:
        descriptor: An instance of cv2.xxxx_create() that will detect and compute
        the descriptors on the frames.

    Returns:
        A numpy array that contains the descriptors stacked together.
    """

    descriptors = []
    for frame in frames:
        computed = compute_raw_descriptors_given_image(frame, descriptor, **kwargs)
        if computed is None:
            continue
        descriptors.append(computed.T)
    if not len(descriptors):
        return None
    stacked = np.column_stack(descriptors)
    return stacked


class CompactDescriptorExtractor:
    def __init__(
        self,
        descriptor,
        factorizer: tp.Optional[mx.MatrixFactorizer] = None,
        raw_top: tp.Optional[int] = 100,
    ):
        self.descriptor = descriptor
        self.factorizer = factorizer
        if factorizer is None:
            self.factorizer = mx.OrthogonalNonnegativeMatrixFactorizer(
                rank=10, rho=0.01, maxiter=100
            )
        self.raw_top = raw_top

    def extract(
        self,
        frames: tp.Generator[np.ndarray, None, None],
        source_id: tp.Optional[str] = None,
    ) -> tp.Generator[dsc.CompactDescriptorVector, None, None]:
        print(f"Extracting descriptors...")
        matrix = compute_raw_descriptors(frames, self.descriptor, raw_top=self.raw_top)
        if matrix is not None:
            left, _ = self.factorizer.factor(matrix)

            for cols in left.T:
                yield dsc.CompactDescriptorVector(vector=cols, source_id=source_id)

    def extract_from_video(
        self,
        grouped_frames: tp.Generator[tp.Generator[np.ndarray, None, None], None, None],
        source_id: tp.Optional[str] = None,
    ) -> tp.Generator[dsc.CompactDescriptorVector, None, None]:
        for gop in grouped_frames:
            yield from self.extract(gop, source_id)

    def save(
        self,
        frames: tp.Generator[np.ndarray, None, None],
        path: tp.Union[str, pathlib.Path],
        source_id: tp.Optional[str] = None,
    ) -> None:
        vectors = [vector.encode() for vector in self.extract(frames, source_id)]
        with open(path, "wb") as f:
            mp.pack({"source_id": source_id, "descriptors": vectors}, f)

    def save_from_video(
        self,
        frames: tp.Generator[tp.Generator[np.ndarray, None, None], None, None],
        path: tp.Union[pathlib.Path, str],
        source_id: tp.Optional[str] = None,
    ) -> None:
        print(f"Extracting descriptors to {path}...")
        vectors = [
            vector.encode() for vector in self.extract_from_video(frames, source_id)
        ]
        with open(path, "wb") as f:
            mp.pack({"source_id": source_id, "descriptors": vectors}, f)

    def save_from_directory(
        self, inp_dir: pathlib.Path, out_dir: pathlib.Path, **kwargs
    ) -> None:

        group_size = kwargs.get("group_size", 30)

        for file in os.listdir(inp_dir):
            inp_src = inp_dir / file
            if os.path.isdir(inp_src):
                self.save_from_directory(inp_src, out_dir, **kwargs)

            mimetype = mimetypes.guess_type(inp_src)[0]
            if mimetype.startswith("video"):
                self.save_from_video(
                    fm.from_video_grouped(inp_src, group_size),
                    out_dir / pathlib.Path(file).with_suffix(".mp"),
                    source_id=pathlib.Path(file).stem,
                )
            elif mimetype.startswith("image"):

                def _wrap(x):
                    yield x

                self.save(
                    _wrap(fm.from_image(inp_src)),
                    out_dir / pathlib.Path(file).with_suffix(".mp"),
                    source_id=pathlib.Path(file).stem,
                )

    def __str__(self) -> str:
        return f"<CompactDescriptorExtractor factorizer={self.factorizer}>"
