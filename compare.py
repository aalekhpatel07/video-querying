import argparse
import pathlib

# from video_onmf import frames as fm
from video_onmf import extractor as ext
from video_onmf import matrix as mx
from video_onmf import descriptor as dsc
from video_onmf import comparator


def create_parser():
    parser = argparse.ArgumentParser(
        description='Match an image/video against the database of descriptors.'
    )
    parser.add_argument(
        'database',
        type=pathlib.Path,
        help="""
        Path to the directory where all '.mp' files are stored.
        """
    )

    parser.add_argument(
        '-f',
        '--file',
        type=pathlib.Path,
        help="""
        The path to the file (ends with .mp) that contains the extracted descriptors
        to match against the database.
        """
    )
    parser.add_argument(
        '-i',
        '--image',
        help='The image to match against the database.',
        type=pathlib.Path,
        default=None
    )
    parser.add_argument(
        '-v',
        '--video',
        help='The video to match against the database.',
        type=pathlib.Path,
        default=None
    )
    parser.add_argument(
        "-t",
        "--top",
        help="""
        The number of top matches to be returned.
        Each match is unique to a source image/video.
        """,
        type = int,
        default = 5
    )

    parser.add_argument(
        "-g",
        "--group_size",
        type=int,
        default=30,
        help="""
        The number of frames to be grouped together for a video so that a
        Group-of-Picture is of this size.

        The default value is 30. As with other stuff, this is subject to experimentation.

        This is irrelevant when extracting compact descriptor from images.
        """
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=30,
        help="""
        The number of cluster centers to store for every Group-of-Pictures for videos,
        or for an image.

        Intuitively, the collection of all the descriptors from some group of pictures (of a video), or just
        one picture (for image), will be compressed to this size. The larger the value,
        the higher the compute time.

        The paper recommends a value of at least 30.
        """
    )

    parser.add_argument(
        "--rho",
        type=float,
        default=.01,
        help="""
        Some fancy hyperparameter described in the Projected Proximal-point
        Alternating Least Squares algorithm in the paper.

        To be honest, I have no idea which value is the best, so I just
        assume the smaller the value the smaller the step in the direction of 
        a better Orthogonal Non-negative Matrix Factorization of the source matrix.
        """
    )

    parser.add_argument(
        "--maxiter",
        type=int,
        default=100,
        help="""
        The maximum number of iterations to perform in order to improve the factorization.

        Again, no idea which value is the best. This is subject to experimentation.
        """
    )

    parser.add_argument(
        "-dt",
        "--descriptor-type",
        type=str,
        default="SIFT",
        help="""
        The type of descriptor to use.

        The paper gives examples using SIFT, but other descriptors like AKAZE, SURF, ORB,
        or BRISK can be used, assuming a corresponding MatrixFactorizer exists.

        For now, we only have SIFT implemented.
        """
    )

    parser.add_argument(
        "-kb",
        "--keep-best",
        type=int,
        default=200,
        help="""
        The number of best (raw descriptors, like SIFT, etc) to keep from each frame.
        The descriptors kept are in the descending order of its keypoint's response.

        The larger the number of descriptors kept, the higher the compute time for factorization.
        We can set a default of 200 but it is subject to experimentation.
        """
    )
    return parser


def setup(rank, rho, iterations, descriptor_type, keep_best):

    factorizer = mx.OrthogonalNonnegativeMatrixFactorizer(
        rank=rank,
        rho=rho,
        maxiter=iterations
    )

    descriptor_ = dsc.NativeDescriptor(descriptor_type).descriptor
    extractor = ext.CompactDescriptorExtractor(
        descriptor=descriptor_,
        factorizer=factorizer,
        raw_top=keep_best
    )

    return extractor


def main():
    parser = create_parser()
    args = parser.parse_args()

    cmp = comparator.CompactDescriptorComparator(args.database)
    results = None

    if args.file is not None:
        results = cmp.match_file(
            args.file,
            top=args.top
        )
    else:
        if all(arg is None for arg in [args.image, args.video]):
            raise ValueError("Either specify an input file with descriptors or at least one of image/video must be given.")

        extractor = setup(
            rank=args.rank,
            rho=args.rho,
            iterations=args.maxiter,
            descriptor_type=args.descriptor_type,
            keep_best=args.keep_best
        )

        if args.image is not None:
            results = cmp.match_image(
                extractor=extractor,
                query=args.image,
                source_id=args.image.stem,
                top=args.top
            )
        elif args.video is not None:
            results = cmp.match_video(
                extractor=extractor,
                query=args.video,
                source_id=args.video.stem,
                top=args.top
            )

    if results is not None:
        print(list(results))


if __name__ == "__main__":
    main()
