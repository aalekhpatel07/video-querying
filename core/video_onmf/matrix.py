import abc
import numpy as np

from typing import (
    Tuple,
)


class MatrixFactorizer(abc.ABC):

    @abc.abstractmethod
    def factor(
        self,
        matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


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

            left_hat = first_term_left @ np.linalg.inv(second_term_left)

            left_positive = np.abs(left_hat)
            left_next = left_positive / np.linalg.norm(left_positive, axis=0)

            # Update right.
            first_term_right = self.rho * np.identity(self.rank) + (left.T @ left)
            second_term_right = self.rho * right + (left_next.T @ matrix)

            left = left_next
            right_final = np.linalg.inv(first_term_right) @ second_term_right

            right = np.zeros_like(right_final)
            right[np.argmax(right_final, axis=0), range(right_final.shape[-1])] = 1

            k += 1

        return left, right

    def __str__(self):
        attrs = []
        for att in ["rank", "rho", "maxiter"]:
            attrs.append(f"{att}={getattr(self, att)}")
        return f"<OrthogonalNonnegativeMatrixFactorizer {', '.join(attrs)}>"
