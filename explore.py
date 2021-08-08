from pathlib import Path
from video_onmf import frames as fm
from video_onmf import extractor as ext
from video_onmf import matrix as mx
from video_onmf import descriptor as dsc


VIDEOS = Path("./data")
PEXELS = VIDEOS / "pexels.mp4"
DOGGIE = VIDEOS / "n02099601_7304.jpg"
BUNNY = VIDEOS / "bunny.mp4"
EARTH = VIDEOS / "earth.mp4"
DESCRIPTORS = VIDEOS / "descriptors"


def main():
    rank = 30
    rho = 0.01
    iterations = 100
    group_size = 30

    factorizer = mx.OrthogonalNonnegativeMatrixFactorizer(
        rank=rank,
        rho=rho,
        maxiter=iterations
    )

    sift = dsc.NativeDescriptor('SIFT').descriptor
    extractor = ext.CompactDescriptorExtractor(
        descriptor=sift,
        factorizer=factorizer
    )

    earth_mp = DESCRIPTORS / "earth.mp"
    gops = fm.from_video_grouped(
        EARTH,
        group_size=group_size
    )

    extractor.save_from_video(gops, earth_mp, 'earth')
    extractor.load(earth_mp)


if __name__ == '__main__':
    main()
