import fire
from pathlib import Path

from video_onmf import frames as fm
from video_onmf import extractor as ext
from video_onmf import matrix as mx
from video_onmf import descriptor as dsc
from video_onmf import comparator

VIDEOS = Path("./data")
PEXELS = VIDEOS / "pexels.mp4"
DOGGIE = VIDEOS / "n02099601_7304.jpg"
SCENERY = VIDEOS / "scenery.jpg"
BUNNY = VIDEOS / "bunny.mp4"
EARTH = VIDEOS / "earth.mp4"
DESCRIPTORS = VIDEOS / "descriptors"


def setup():
    rank = 30
    rho = 0.01
    iterations = 100

    factorizer = mx.OrthogonalNonnegativeMatrixFactorizer(
        rank=rank,
        rho=rho,
        maxiter=iterations
    )

    sift = dsc.NativeDescriptor('SIFT').descriptor
    extractor = ext.CompactDescriptorExtractor(
        descriptor=sift,
        factorizer=factorizer,
    )

    return extractor


extractor = setup()


def extract_and_save():

    # group_size = 30
    file_mp = DESCRIPTORS / "scenery.mp"
    gops = fm.from_image(SCENERY)

    def temp(x):
        yield x
    
    # gops = fm.from_video_grouped(
    #     BUNNY,
    #     group_size=group_size
    # )
    extractor.save(temp(gops), file_mp, 'scenery')
    return


# def load_descriptors_of_single_file(file_path):
#     return comparator.load_file(file_path)

def temp(vid):
    yield vid


def main():
    # extract_and_save()

    cmp = comparator.CompactDescriptorComparator(DESCRIPTORS)
    res = cmp.match_file(DESCRIPTORS / "scenery.mp", top=2)
    print(res)
    # res = cmp.match_image(
    #     extractor=extractor,
    #     query=SCENERY,
    #     top=10
    # )
    # print(res)
    # vid = fm.from_video_grouped(EARTH, group_size=10)
    # vla = extractor.extract_from_video(vid)
    # for x in vla:
    #     # print(type(x))
    #     # res = cmp.compare(x)
    #     print("GOP")
    #     for res in cmp._match_descriptor(x, top=5):
    #         print(res)
            # print(res)
        # print(x)
    # extractor.extract()
    # for d in load_descriptors_of_single_file(DESCRIPTORS / "earth.mp"):
    #     print(d)
    # print(cmp._compare())
    # print(load_descriptors_of_single_file(DESCRIPTORS / "earth.mp"))
    pass


if __name__ == '__main__':
    main()
