import datetime
import time

from video_onmf import algorithms as alg
import numpy as np
from pathlib import Path
import pandas as pd


VIDEOS = Path("./data")
PEXELS = VIDEOS / "pexels.mp4"

# def main():

    # for idx, group in enumerate(alg.get_groups_of_pictures("./data/bunny.mp4")):
    #     descriptors_gop = alg.combine_descriptors_per_group_of_picture(group, sift)
    #     left, right = alg.extract_compact_descriptors_from_group(descriptors_gop)
    #     print(left.shape, left)
    #     break
    #
    # video_stream_grouped = alg.get_groups_of_pictures("./data/pexels-tima-miroshnichenko-5377684.mp4", group_size=30)
    # alg.extract_compact_descriptors_into_directory(
    #     video_stream_grouped,
    #     video_name='bunny',
    #     output_dir=Path("./data/descriptors"),
    #     sift=sift
    # )

    # matrix = np.random.random(size=(128, 80))
    # left, right = alg.pppals(matrix, 30, 1, 500)
    # prod = left @ right
    # diff = np.abs(matrix - prod)
    # print(alg.norm(diff))
    # noise_min(matrix, 30)

    # return


def noise_min(matrix, rank, number=1000):
    val = np.inf
    for _ in range(number):
        left, right = np.random.random(size=(128, rank)), np.random.randint(2, size=(rank, 43313))
        prod = left @ right
        temp = np.linalg.norm(np.abs(matrix - prod))
        if temp < val:
            print(f"improved: ", temp)
            val = temp
    return val

def main():
    video_capture = alg.capture_video(PEXELS)
    video_id = "pexels"

    frames_per_group = 30

    compact_descriptors_per_group = 30
    some_fancy_hyperparameter_rho = 0.01
    max_number_of_iterations_for_refinement = 100

    vid = alg.VideoIntoGroupedPictures(video_capture,
                                       video_id=video_id,
                                       group_size=frames_per_group
                                       )
    factorizer = alg.OrthogonalNonnegativeMatrixFactorizer(
        rank=compact_descriptors_per_group,
        rho=some_fancy_hyperparameter_rho,
        maxiter=max_number_of_iterations_for_refinement
    )
    with alg.Descriptor('SIFT') as d:
        extractor = alg.CompactDescriptorExtractor(descriptor=d,
                                                   factorizer=factorizer
                                                   )
        extractor.save_from_video(vid, VIDEOS / "descriptors")
        # it = iter(vid.group_of_pictures)
        # next(it)
        # gop = next(it)
        # extractor.save(gop, VIDEOS / f"descriptors/{gop.gop_id}.mp")
        # print(gop)
        # cd, gop1 = extractor.load(VIDEOS / "descriptors/bunny_001.mp")
        # print(cd)
        # print(gop1)
        # print(gop == gop1)
        # print(cd)
        # print(gop)
        # print(gop)
        # extractor.save(gop, VIDEOS / f"descriptors/{gop.gop_id}.mp")
        # extractor.
        # for cdesc in vid.get_compact_descriptors_from(extractor):
        #     start = datetime.datetime.now()
        #     alg.CompactDescriptor.decode(cdesc.encode())
        #     end = datetime.datetime.now()
        #     print((end - start).total_seconds())
        #     break

    # print(vid)
    # for elem in vid.group_of_pictures:
    #     print(elem)

    # extractor = alg.CompactDescriptorExtractor()
    # with alg.Descriptor('SIFT') as desc_type:
    #     extractor.extract()
    # extractor = alg.CompactDescriptorExtractor(
    #     factorizer=alg.OrthogonalNonnegativeMatrixFactorizer(rank=10, rho=.01, maxiter=1000)
    # )
    # with alg.Descriptor('SIFT') as desc_type:
    #     compact_descriptor = extractor.extract(gop, desc_type)
    return


if __name__ == '__main__':
    main()
    # print(extractor)
    # extractor.extract()
    # pexels_frame_sep = alg.VideoIntoGroupedPictures(
    #     video_capture=video_capture,
    #     group_size=50
    # )
    # print(next(iter(pexels_frame_sep.group_of_pictures)))
    #
    # with alg.Descriptor('SIFT') as descriptor_instance:
    #     for idx, it in enumerate(pexels_frame_sep.group_of_pictures):
    #         desc = it.compute_raw_descriptors(descriptor_instance)
    #         print(desc.shape)
    #         break
