import abc
# import functools
import os
from typing import (
    Tuple,
    Dict,
    Generator,
    Union,
    Iterable,
    Optional,
    Literal,
    Any
)
# from itertools import (
#     groupby,
#     count,
#     repeat,
#     zip_longest,
#     tee
# )
import math
# from functools import partial
# from pathlib import Path
# import numpy as np
# from numpy.linalg import (
#     inv,
#     norm
# )
import numpy as np
import numpy.linalg

# from matplotlib import pyplot as plt
# from numba import (
#     jit,
#     njit,
#     vectorize
# )
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Pool
import cv2 as cv
from .utils import *
# from math import inf
import msgpack as mp
import msgpack_numpy as mnp


class VideoUnReadable(Exception):
    """
    Indicates that a video file could not be read by cv2.
    """
    pass


class MatrixFactorizer(abc.ABC):

    @abc.abstractmethod
    def factor(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass


class Descriptor:
    def __init__(self, descriptor_type: str = 'SIFT', *args, **kwargs):
        self.type = descriptor_type
        self.descriptor = None

        if not hasattr(cv, f'{descriptor_type}_create'):
            raise NotImplementedError(f"OpenCV does not understand {descriptor_type}_create()")

        descriptor_ = getattr(cv, f'{descriptor_type}_create', None)
        if descriptor_ is None or not callable(descriptor_):
            raise NotImplementedError(f"OpenCV does not understand {descriptor_type}_create()")

        if descriptor_type != 'SIFT':
            raise NotImplementedError("Only SIFT is supported as of yet.")

        self.descriptor = descriptor_(*args, **kwargs)

    def __enter__(self):
        return self.descriptor

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.descriptor


class CompactDescriptor:

    left_shape: Tuple
    right_shape: Tuple
    gop_id: str

    def __init__(self,
                 left: np.ndarray,
                 right: Optional[np.ndarray] = None,
                 gop_id: Optional[str] = None,
                 ):
        self.left = left
        self.right = right
        self.gop_id = gop_id

        self.left_shape = self.left.shape
        if self.right is not None:
            self.right_shape = self.right.shape
        pass

    def __str__(self):
        attrs = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"<CompactDescriptor {', '.join(attrs)}>"

    @property
    def __dict__(self):
        return {
            'left_shape': self.left.shape,
            'right_shape': self.right and self.right.shape,
            'gop_id': self.gop_id
        }

    @property
    def vectors(self):
        return self.left.T

    def encode(self):
        left_enc = mp.packb(self.left.T, default=mnp.encode)
        # right_enc = mp.packb(self.right.T, default=mnp.encode)
        return mp.packb({
            'left_enc': left_enc,
            # 'right_enc': right_enc,
            'gop_id': self.gop_id
        })

    @classmethod
    def decode(cls, obj):
        dec = mp.unpackb(obj)
        left_dec = mp.unpackb(dec["left_enc"], object_hook=mnp.decode)
        # right_dec = mp.unpackb(dec["right_enc"], object_hook=mnp.decode)
        gop_id = dec["gop_id"]

        return cls(left=left_dec,
                   # right=right_dec,
                   gop_id=gop_id
                   )

    def compare(self):
        return

    def __eq__(self, other):
        if not isinstance(other, CompactDescriptor):
            return False
        return np.allclose(self.left, other.left)


class VideoIntoGroupedPictures:
    def __init__(self,
                 video_capture: cv.VideoCapture,
                 video_channels: int = 3,
                 group_size: int = 30,
                 video_id: Optional[str] = None
                 ):
        metadata = probe(video_capture)

        frame_stream = stream(video_capture)
        gops_raw = grouper(frame_stream,
                           group_size,
                           )
        gops_clean = map(np.array, gops_raw)

        self.video_metadata = metadata
        self.video_channels = video_channels
        self.group_size = group_size
        self.total_gops = math.ceil(metadata['total_frames'] / self.group_size)
        self.video_id = video_id
        self.group_of_pictures: Generator[GroupOfPictures, None, None] = (
            GroupOfPictures(video_capture=None,
                            arr=gop,
                            video_start_frame=idx * self.group_size,
                            video_end_frame=idx * self.group_size + len(gop),
                            video_id=video_id,
                            gop_id=video_id and f"{video_id}_{idx:03d}"
                            ) for idx, gop in enumerate(gops_clean)
        )

    @property
    def __dict__(self):
        return {
            'video_id': self.video_id,
            'group_size': self.group_size,
            'video_channels': self.video_channels,
            'total_gops': self.total_gops,
            **self.video_metadata
        }

    def __str__(self):
        attrs = [f"{k}={v}" for k, v in self.__dict__.items()]
        s = f"<VideoIntoGroupedPictures {' '.join(attrs)}>"
        return s


class GroupOfPictures(np.ndarray):
    def __new__(cls,
                video_start_frame: int,
                video_end_frame: int,
                video_capture: Optional[cv.VideoCapture] = None,
                arr: Optional[np.ndarray] = None,
                gop_id: Optional[str] = None,
                video_id: Optional[str] = None
                ):

        # Important if the array is pre-calculated (as in VideoIntoGroupedPictures).
        if arr is not None:
            obj = np.asarray(arr).view(cls)
        elif video_capture is not None:
            obj = np.asarray(list(stream(video_capture, start=video_start_frame, end=video_end_frame))).view(cls)
        else:
            obj = np.asarray([]).view(cls)
        obj.video_start_frame = video_start_frame
        obj.video_end_frame = video_end_frame
        obj.video_id = video_id
        obj.gop_id = gop_id

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.video_start_frame = getattr(obj, 'video_start_frame', None)
        self.video_end_frame = getattr(obj, 'video_end_frame', None)
        self.video_id = getattr(obj, 'video_id', None)
        self.gop_id = getattr(obj, 'gop_id', None)

    @property
    def __dict__(self):
        return {
            'start_frame': self.video_start_frame,
            'end_frame': self.video_end_frame,
            'video_id': self.video_id,
            'gop_id': self.gop_id
        }

    def __str__(self):
        attrs = [f"{k}={v}" for k, v in self.__dict__.items()]
        s = f"<GroupOfPictures {', '.join(attrs)}>"
        return s

    def compute_raw_descriptors(self, descriptor) -> np.ndarray:
        """Compute and stack the descriptors from all the frames within the group.

        Args:
            descriptor: An instance of cv2.xxxx_create() that will detect and compute
            the descriptors on the frames.

        Returns:
            A numpy array that contains the descriptors stacked together.
        """

        descriptors = []
        for frame in self:
            computed = compute_raw_descriptors_given_image(frame, descriptor)
            if computed is None:
                continue
            descriptors.append(computed.T)

        stacked = np.column_stack(descriptors)
        return stacked

    def __eq__(self, other):
        if not isinstance(other, GroupOfPictures):
            return False

        if self.gop_id == other.gop_id and self.video_id == other.video_id:
            return True
        if other.shape != self.shape:
            return False
        return np.allclose(self, other)


def compute_raw_descriptors_given_image(img: np.ndarray, descriptor) -> Optional[np.ndarray]:
    """Compute the given descriptors in a given image.

    Args:
        img: A numpy array representation of an image.
        descriptor: The descriptor type to extract.

    Returns:
        The extracted descriptors.
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kp, des = descriptor.detectAndCompute(gray, None)
    if des is None or not len(des):
        return None
    des /= numpy.linalg.norm(des, axis=0)
    return des


class CompactDescriptorExtractor:
    def __init__(self,
                 descriptor,
                 factorizer: Optional[MatrixFactorizer] = None,
                 ):
        self.descriptor = descriptor
        self._factorizer = factorizer
        if factorizer is None:
            self._factorizer = OrthogonalNonnegativeMatrixFactorizer(
                rank=10,
                rho=.01,
                maxiter=100
            )

    def extract(self, gop: GroupOfPictures) -> CompactDescriptor:
        print(f"Extracting gop: {gop.gop_id} ...")
        matrix = gop.compute_raw_descriptors(self.descriptor)
        left, right = self._factorizer.factor(matrix)
        print(f"Extraction complete for gop: {gop.gop_id} ...")
        return CompactDescriptor(left, right, gop_id=gop.gop_id)

    def save(self, gop: GroupOfPictures, path: Union[str, Path]) -> None:
        print(f"Extracting descriptors from {gop.gop_id}...")
        compact_descriptor = self.extract(gop)
        print(f"Writing descriptors for {gop.gop_id}...")
        to_dump = {
            'descriptor': compact_descriptor.encode(),
            **gop.__dict__
        }
        with open(Path(path), 'wb') as f:
            mp.dump(to_dump, f)
        print("Saved successfully!")

    def extract_from_video(self,
                           video: VideoIntoGroupedPictures
                           ) -> Generator[CompactDescriptor, None, None]:
        for gop in video.group_of_pictures:
            yield self.extract(gop)

    def save_from_video(self,
                        video: VideoIntoGroupedPictures,
                        directory: Union[Path, str],
                        ) -> None:
        print(f"Extracting descriptors from video in {directory}...")

        def _save(x: GroupOfPictures):
            self.save(x, directory / f"{video.video_id}_{x.gop_id}.mp")

        for gop in video.group_of_pictures:
            _save(gop)
        # process_pool = Pool(processes=1)
        # process_pool.map(_save, video.group_of_pictures)

        #
        # descriptors = [
        #     descriptor.encode() for descriptor in self.extract_from_video(video)
        # ]
        # print(f"Writing descriptors to file: {path}")
        # to_dump = {
        #     'descriptors': descriptors,
        #     **video.__dict__
        # }
        # with open(path, 'wb') as f:
        #     mp.dump(to_dump, f)
        # print("Saved successfully.")

    @classmethod
    def load(cls, path: Union[str, Path]) -> Tuple[CompactDescriptor, GroupOfPictures]:
        with open(path, 'rb') as f:
            obj = mp.unpack(f)
            descriptor = CompactDescriptor.decode(obj['descriptor'])
            gop = GroupOfPictures(video_start_frame=obj['start_frame'],
                                  video_end_frame=obj['end_frame'],
                                  video_id=obj['video_id'],
                                  gop_id=obj['gop_id']
                                  )
            return descriptor, gop

    @classmethod
    def load_from_video(cls, path: Union[Path, str]):

        return

    def __str__(self):
        return f"<CompactDescriptorBuilder factorizer={self._factorizer}>"


class OrthogonalNonnegativeMatrixFactorizer(MatrixFactorizer):
    """A Solver for ONMF.

    """

    def __init__(self, rank: int, rho: float, maxiter: int):
        """Construct a Solver for ONMF.

        Args:
            rank: The desired number of columns in L.
            rho: Some hyper-parameter that I don't understand yet.
            maxiter: The maximum number of iterations of improvement.
        """
        self.rank = rank
        self.rho = rho
        self.maxiter = maxiter

    def factor(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The Projected Proximal Point Alternating Least Squares algorithm.

        Notes:
            This algorithm is called "Projected
            Proximal-point Alternating Least Squares"
            and is introduced in "Video Querying Via
            Compact Descriptors Of Visually Salient
            Objects".

            Given a matrix M of dimensions (m, N),
            factor it into two matrices L, and R, such that:

            - M ≈ L @ R,
            - The columns of L are unit vectors,
            - Every column of R has exactly one 1, and the rest 0,
            - The (Euclidean) distance between M and L @ R is minimized amongst
                all possible L, and R with non-negative entries.


        Args:
            matrix: A numpy array of shape (m, N).

        Returns:
            Matrices L, and R that satisfy the conditions given above.
        """

        feature_vector_size, total_vectors = matrix.shape

        k = 0

        left: np.ndarray = np.random.random(size=(feature_vector_size, self.rank))
        right: np.ndarray = np.random.random(size=(self.rank, total_vectors))

        while k < self.maxiter:
            # Update left.
            first_term_left = self.rho * left + (matrix @ right.T)
            second_term_left = (self.rho * np.identity(self.rank)) + (right @ right.T)

            left_hat = first_term_left @ numpy.linalg.inv(second_term_left)

            left_positive = np.abs(left_hat)
            left_next = left_positive / numpy.linalg.norm(left_positive, axis=0)

            # Update right.
            first_term_right = self.rho * np.identity(self.rank) + (left.T @ left)
            second_term_right = self.rho * right + (left_next.T @ matrix)

            left = left_next
            right_final = numpy.linalg.inv(first_term_right) @ second_term_right

            right = np.zeros_like(right_final)
            right[np.argmax(right_final, axis=0), range(right_final.shape[-1])] = 1

            k += 1

        return left, right

    def __str__(self):
        attrs = []
        for att in ["rank", "rho", "maxiter"]:
            attrs.append(f"{att}={getattr(self, att)}")
        return f"<OrthogonalNonnegativeMatrixFactorizer {', '.join(attrs)}>"


def probe(video: cv.VideoCapture) -> Dict[str, Union[int, float, bool]]:
    if not video.isOpened():
        raise VideoUnReadable()

    metadata = {
        "width": int(video.get(cv.CAP_PROP_FRAME_WIDTH)),
        "height": int(video.get(cv.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(video.get(cv.CAP_PROP_FPS)),
        "total_frames": int(video.get(cv.CAP_PROP_FRAME_COUNT)),
        "convert_to_rgb": bool(video.get(cv.CAP_PROP_CONVERT_RGB)),
    }

    return metadata


def capture_video(video_path: Union[str, Path]):
    return cv.VideoCapture(str(video_path))


def stream(video_stream: cv.VideoCapture, **kwargs) -> Generator[np.ndarray, None, None]:
    """Generate a stream of frames from a video.

    If start, or end are specified then frames
    with index [start, end) are streamed instead
    of the entire video.

    Args:
        video_stream: The open video stream.
        **kwargs: A dict of optional arguments.

    Yields:
        Consecutive frames from the video.
    """

    metadata = probe(video_stream)

    start = kwargs.get('start', 0)
    end = kwargs.get('end', metadata.get('total_frames') + 1)

    ctr = 0
    while video_stream.isOpened():
        is_good, frame = video_stream.read()
        ctr += 1
        if is_good and start <= ctr < end:
            yield frame
        if ctr >= end:
            break


def grouper(iterable, n):
    """Collect data into fixed-length chunks or blocks"""
    # args = list(tee(iter(iterable), n))
    current = []
    while True:
        try:
            current.append(next(iterable))
            if len(current) % n == 0:
                yield current
                current = []
        except StopIteration:
            if len(current):
                yield current
            break


# def groups_of_pictures(frame_stream: Iterable[np.ndarray],
#                        width: int,
#                        height: int,
#                        group_size: int = 30
#                        ) -> Generator[np.ndarray, None, None]:
#     """
#
#     Args:
#         frame_stream:
#         width:
#         height:
#         group_size:
#
#     Returns:
#
#     """
#
#     yield from map(np.array,
#                    grouper(frame_stream,
#                            group_size,
#                            fillvalue=np.empty(shape=(height, width))
#                            )
#                    )


# ufunc_compute_descriptor = np.frompyfunc(compute_sift_descriptors, 1, 1)


#
# def compute_akaze_descriptors(img: np.ndarray, sift):
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     kp, des = sift.detectAndCompute(gray, None)
#     des /= norm(des, axis=0)
#     if des is None or not len(des):
#         return None
#     return des


# def combine_descriptors_per_group_of_picture(gop, sift):
#     arr = []
#     for img in gop:
#         val = compute_descriptors(img, sift)
#         if val is not None:
#             arr.append(val.T)
#     return np.column_stack(arr)


# def projected_proximal_point_alternating_least_squares(
#         matrix: np.ndarray,
#         rank: int = 50,
#         rho: float = 1,
#         maxiter: int = 100
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Compute the Orthogonal Non-negative Matrix Factorization.
#
#     Notes:
#         This algorithm is called "Projected
#         Proximal-point Alternating Least Squares"
#         and is introduced in "Video Querying Via
#         Compact Descriptors Of Visually Salient
#         Objects".
#
#         Given a matrix M of dimensions (m, N),
#         factor it into two matrices L, and R, such that:
#
#         - M ≈ L @ R,
#         - The columns of L are unit vectors,
#         - Every column of R has exactly one 1, and the rest 0,
#         - The (Euclidean) distance between M and L @ R is minimized amongst
#             all possible L, and R with non-negative entries.
#
#
#     Args:
#         matrix: A numpy array of shape (m, N).
#         rank: The desired number of columns in L.
#         rho: Some hyper-parameter that I don't understand yet.
#         maxiter: The maximum number of iterations of improvement.
#
#     Returns:
#         Matrices L, and R that satisfy the conditions given above.
#     """
#
#     feature_vector_size, total_vectors = matrix.shape
#
#     k = 0
#
#     left: np.ndarray = np.random.random(size=(feature_vector_size, rank))
#     right: np.ndarray = np.random.random(size=(rank, total_vectors))
#
#     while k < maxiter:
#         # Update left.
#         first_term_left = rho * left + (matrix @ right.T)
#         second_term_left = (rho * np.identity(rank)) + (right @ right.T)
#
#         left_hat = first_term_left @ inv(second_term_left)
#
#         left_positive = np.abs(left_hat)
#         left_next = left_positive / norm(left_positive, axis=0)
#
#         # Update right.
#         first_term_right = rho * np.identity(rank) + (left.T @ left)
#         second_term_right = rho * right + (left_next.T @ matrix)
#
#         left = left_next
#         right_final = inv(first_term_right) @ second_term_right
#
#         right = np.zeros_like(right_final)
#         right[np.argmax(right_final, axis=0), range(right_final.shape[-1])] = 1
#
#         k += 1
#
#     return left, right
#
#
# def pppals(*args, **kwargs):
#     return projected_proximal_point_alternating_least_squares(*args, **kwargs)


# def extract_compact_descriptors_from_group(group, *args, **kwargs):
#     left, right = projected_proximal_point_alternating_least_squares(
#         matrix=group,
#         *args,
#         **kwargs
#     )
#     return left, right


def extract_compact_descriptors_into_directory(
        frames_grouped_stream,
        video_name: str,
        output_dir: Path,
        sift,
        *args,
        **kwargs
):
    for idx, gop in enumerate(frames_grouped_stream):
        descriptors_gop = combine_descriptors_per_group_of_picture(gop, sift)
        print(f"Extracting compact descriptors for group {idx:03d}")
        left, right = extract_compact_descriptors_from_group(descriptors_gop, *args, **kwargs)
        print(f"Saving compact descriptors for group {idx:03d}")
        save_dir = output_dir / f'{video_name}_{idx:03d}.npy'
        with open(save_dir, 'wb') as f:
            np.save(f, left)
        break
    print(f"Saved to {output_dir}/{video_name}_*.npy successfully!")
    return


class VideoDescriptors:

    @classmethod
    def from_directory(cls):
        return

    def __init__(self):
        pass
