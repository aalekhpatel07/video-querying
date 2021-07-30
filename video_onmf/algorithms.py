from typing import (
    Tuple
)
from pathlib import Path
import numpy as np
from numpy.linalg import (
    inv,
    norm
)
from matplotlib import pyplot as plt
from numba import (
    jit,
    njit,
    vectorize
)
from concurrent.futures import ThreadPoolExecutor
import cv2 as cv


def get_groups_of_pictures(video_path: str, group_size: int = 30):
    video_stream = cv.VideoCapture(video_path)
    coll = []
    while video_stream.isOpened():
        is_good, frame = video_stream.read()
        if not is_good:
            yield coll
            coll = []
            break
        if len(coll) < group_size:
            coll.append(frame)
        else:
            yield coll
            coll = []
    if len(coll):
        yield coll


def compute_sift_descriptors(img):
    return


def projected_proximal_point_alternating_least_squares(
    matrix: np.ndarray,
    rank: int,
    rho: float,
    maxiter: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the Orthogonal Non-negative Matrix Factorization.

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
        rank: The desired number of columns in L.
        rho: Some hyper-parameter that I don't understand yet.
        maxiter: The maximum number of iterations of improvement.

    Returns:
        Matrices L, and R that satisfy the conditions given above.
    """

    feature_vector_size, total_vectors = matrix.shape

    k = 0

    left: np.ndarray = np.random.random(size=(feature_vector_size, rank))
    right: np.ndarray = np.random.random(size=(rank, total_vectors))

    while k < maxiter:
        # Update left.
        first_term_left = rho * left + (matrix @ right.T)
        second_term_left = (rho * np.identity(rank)) + (right @ right.T)

        left_hat = first_term_left @ inv(second_term_left)

        left_positive = np.abs(left_hat)
        left_next = left_positive / norm(left_positive, axis=0)

        # Update right.
        first_term_right = rho * np.identity(rank) + (left.T @ left)
        second_term_right = rho * right + (left_next.T @ matrix)

        left = left_next
        right_final = inv(first_term_right) @ second_term_right

        right = np.zeros_like(right_final)
        right[np.argmax(right_final, axis=0), range(right_final.shape[-1])] = 1

        k += 1

    return left, right


def pppals(*args, **kwargs):
    return projected_proximal_point_alternating_least_squares(*args, **kwargs)
