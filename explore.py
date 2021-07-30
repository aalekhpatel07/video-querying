from video_onmf import algorithms as alg
import numpy as np
from pathlib import Path

VIDEOS = Path("./data")


def main():

    for idx, group in enumerate(alg.get_groups_of_pictures("./data/bunny.mp4")):
        print(idx)

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
