import glob
import collections as coll
import typing as tp
import os
import pathlib
import multiprocessing as mlp
import functools
import heapq
import operator as op
import numpy as np
import msgpack as mp

from . import extractor as ext
from . import descriptor as dsc
from . import frames as fm


def _get_mp(
    directory: tp.Union[pathlib.Path, str]
) -> tp.Generator[pathlib.Path, None, None]:
    root = pathlib.Path(directory)
    yield from root.rglob("*.mp")


def load_file(
    file_path: pathlib.Path,
) -> tp.Tuple[str, tp.Generator[dsc.CompactDescriptorVector, None, None]]:
    with open(file_path, "rb") as f:
        obj = mp.unpack(f)
        descriptors = (
            dsc.CompactDescriptorVector.decode(x) for x in obj["descriptors"]
        )
    return obj["source_id"], descriptors


def load_database(
    directory: tp.Union[pathlib.Path, str],
) -> tp.Tuple[str, tp.Generator[dsc.CompactDescriptorVector, None, None]]:
    """"""
    yield from map(load_file, _get_mp(directory))


class CompactDescriptorComparator:
    def __init__(
        self,
        database_root: tp.Union[pathlib.Path, str],
        compare_fn: tp.Callable[[np.ndarray, np.ndarray], np.float] = np.dot,
    ):
        self.compare_fn = compare_fn
        self.database_root = pathlib.Path(database_root)
        self.order_fn = functools.partial(max, key=np.abs)

    def match_descriptor(
        self, descriptor: dsc.CompactDescriptorVector, **kwargs
    ) -> tp.List[tp.Tuple[float, str]]:
        """"""
        top = kwargs.get("top", 1)
        heap = []

        for source_id, stored_descriptors in load_database(self.database_root):
            # Let's go for parallelizing search within each file.
            # With multiprocessing...

            # with mlp.Pool(processes=os.cpu_count()) as pool:
            #     matches = pool.map(
            #         functools.partial(self.compare_fn, descriptor),
            #         stored_descriptors
            #     )

            # Without multiprocessing...
            # with mlp.Pool(processes=os.cpu_count()) as pool:
            matches = list(
                map(functools.partial(self.compare_fn, descriptor), stored_descriptors)
            )

            score = self.order_fn(matches or [0])
            heapq.heappush(heap, (score, source_id))

        return heapq.nlargest(n=top, iterable=heap)

    def match_video(
        self,
        extractor: ext.CompactDescriptorExtractor,
        query: tp.Union[pathlib.Path, str],
        source_id: tp.Optional[str] = None,
        **kwargs,
    ) -> tp.List[tp.Tuple[str, float]]:

        top = kwargs.get("top", 1)
        group_size = kwargs.get("group_size", 30)

        vectors = extractor.extract_from_video(
            fm.from_video_grouped(query, group_size), source_id=source_id
        )

        best = coll.defaultdict(lambda: 0)
        for descriptor in vectors:
            for (score, source_id) in self.match_descriptor(descriptor, top=top):
                best[source_id] = max(score, best[source_id])

        return heapq.nlargest(n=top, iterable=best.items(), key=op.itemgetter(1))

    def match_image(
        self,
        extractor: ext.CompactDescriptorExtractor,
        query: tp.Union[pathlib.Path, str],
        source_id: tp.Optional[str] = None,
        **kwargs,
    ) -> tp.List[tp.Tuple[str, float]]:

        top = kwargs.get("top", 1)
        # group_size = kwargs.get('group_size', 30)

        def _wrap(x):
            yield x

        vectors = extractor.extract(_wrap(fm.from_image(query)), source_id=source_id)

        best = coll.defaultdict(lambda: 0)
        for descriptor in vectors:
            for (score, source_id) in self.match_descriptor(descriptor, top=top):
                best[source_id] = max(score, best[source_id])

        return heapq.nlargest(n=top, iterable=best.items(), key=op.itemgetter(1))

    def match_file(
        self, query: pathlib.Path, **kwargs
    ) -> tp.List[tp.Tuple[str, float]]:

        top = kwargs.get("top", 1)

        def _wrap(x):
            yield x

        best = coll.defaultdict(lambda: 0)
        query_id, stored_descriptors = next(_wrap(load_file(query)))

        print(
            f"Matching {query.stem}{query.suffix} (stored as '{query_id}') against the database..."
        )

        for descriptor in stored_descriptors:
            for (score, source_id) in self.match_descriptor(descriptor, top=top):
                best[source_id] = max(score, best[source_id])

        return heapq.nlargest(n=top, iterable=best.items(), key=op.itemgetter(1))
