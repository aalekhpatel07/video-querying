import cv2.xfeatures2d

from video_onmf import algorithms as alg
import numpy as np
from pathlib import Path
import pandas as pd


VIDEOS = Path("./data")

sift = cv2.xfeatures2d.SIFT_create()

def main():

    # for idx, group in enumerate(alg.get_groups_of_pictures("./data/bunny.mp4")):
    #     descriptors_gop = alg.combine_descriptors_per_group_of_picture(group, sift)
    #     left, right = alg.extract_compact_descriptors_from_group(descriptors_gop)
    #     print(left.shape, left)
    #     break
    #
    video_stream_grouped = alg.get_groups_of_pictures("./data/bunny.mp4", group_size=30)
    alg.extract_compact_descriptors_into_directory(
        video_stream_grouped,
        video_name='bunny',
        output_dir=Path("./data/descriptors"),
        sift=sift
    )

    # matrix = np.random.random(size=(128, 80))
    # left, right = alg.pppals(matrix, 30, 1, 500)
    # prod = left @ right
    # diff = np.abs(matrix - prod)
    # print(alg.norm(diff))
    # noise_min(matrix, 30)

    return


def noise_min(matrix, rank, number=1000):
    val = np.inf
    for _ in range(number):
        left, right = np.random.random(size=(128, rank)), np.random.random(size=(rank, 80))
        prod = left @ right
        temp = alg.norm(np.abs(matrix - prod))
        if temp < val:
            print(f"improved: ", temp)
            val = temp
    return val


if __name__ == '__main__':
    main()
