# import typing as tp
# import functools
# import numpy as np
# from . import image
#
#
# def _extend(func):
#     @functools.wraps(func)
#     def from_frame_stream(
#         frames: tp.Generator[np.ndarray, None, None], *args, **kwargs
#     ) -> tp.Generator[np.ndarray, None, None]:
#         for frame in frames:
#             yield func(frame, *args, **kwargs)
#
#     return from_frame_stream
#
#
# flip_horizontal = _extend(image.flip_horizontal)
# flip_vertical = _extend(image.flip_vertical)
# color_filter = _extend(image.color_filter)
# pillarbox = _extend(image.pillarbox)
# letterbox = _extend(image.letterbox)
# windowbox = _extend(image.windowbox)
# rotate = _extend(image.rotate)
# crop = _extend(image.crop)
# scale = _extend(image.scale)
# scale_uniform = _extend(image.scale_uniform)
# scale_to_fit = _extend(image.scale_to_fit)
