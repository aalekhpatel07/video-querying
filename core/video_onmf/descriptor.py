import typing as tp
import pathlib
import numpy as np
import cv2 as cv
import msgpack as mp
import msgpack_numpy as mnp


class NativeDescriptor:
    """ """

    def __init__(self, descriptor_type: str = "SIFT", *args, **kwargs):
        self.type = descriptor_type
        self.descriptor = None

        if not hasattr(cv, f"{descriptor_type}_create"):
            raise NotImplementedError(
                f"OpenCV does not understand {descriptor_type}_create()"
            )

        descriptor_ = getattr(cv, f"{descriptor_type}_create", None)
        if descriptor_ is None or not callable(descriptor_):
            raise NotImplementedError(
                f"OpenCV does not understand {descriptor_type}_create()"
            )

        # if descriptor_type != 'SIFT':
        #    raise NotImplementedError("Only SIFT is supported as of yet.")

        self.descriptor = descriptor_(*args, **kwargs)


class CompactDescriptor:
    """ """

    left_shape: tp.Tuple
    right_shape: tp.Tuple
    source_id: str

    def __init__(
        self,
        left: np.ndarray,
        right: tp.Optional[np.ndarray] = None,
        source_id: tp.Optional[str] = None,
    ):
        self.left = left
        self.right = right
        self.source_id = source_id

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
            "left_shape": self.left.shape,
            "right_shape": self.right and self.right.shape,
            "source_id": self.source_id,
        }

    def encode(self):
        left_enc = mp.packb(self.left, default=mnp.encode)
        # right_enc = mp.packb(self.right.T, default=mnp.encode)
        return mp.packb(
            {
                "left_enc": left_enc,
                # 'right_enc': right_enc,
                "source_id": self.source_id,
            }
        )

    @classmethod
    def decode(cls, obj):
        dec = mp.unpackb(obj)
        left_dec = mp.unpackb(dec["left_enc"], object_hook=mnp.decode)
        # right_dec = mp.unpackb(dec["right_enc"], object_hook=mnp.decode)
        source_id = dec["source_id"]

        return cls(
            left=left_dec,
            # right=right_dec,
            source_id=source_id,
        )

    def __eq__(self, other):
        if not isinstance(other, CompactDescriptor):
            return False
        return np.allclose(self.left, other.left)


class CompactDescriptorVector(np.ndarray):
    """A vector representation of a compact video descriptor."""

    def __new__(cls, vector: np.ndarray, source_id: tp.Optional[str] = None):
        obj = np.asarray(vector).view(cls)
        obj.source_id = source_id
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.source_id = getattr(obj, "source_id", None)

    def encode(self):
        return mp.packb(self, default=mnp.encode)

    @classmethod
    def decode(cls, obj):
        return cls(vector=mnp.unpackb(obj, object_hook=mnp.decode), source_id=None)
