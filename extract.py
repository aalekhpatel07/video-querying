import argparse
import pathlib
import os
import mimetypes

from core.video_onmf import frames as fm
from core.video_onmf import extractor as ext
from core.video_onmf import matrix as mx
from core.video_onmf import descriptor as dsc


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        type=pathlib.Path,
        help="""
        Path to input image/video or directory.
        
        If the input is a video, then its compact descriptors will be extracted
        and stored in [out].

        If the input is an image, then its compact descriptors will be extracted
        and stored in [out].

        If the input is a directory, then the compact descriptors for all the media files
        in the directory will be extracted and stored in [out].
        """
    )
    parser.add_argument(
        "output",
        type=pathlib.Path,
        help="""
        Path to output image/video or directory.
        
        If the input is a video, then its compact descriptors will be extracted
        and stored in [out].mp

        If the input is an image, then its compact descriptors will be extracted
        and stored in [out].mp

        If the input is a directory, then the compact descriptors for all the media files
        in the directory will be extracted and stored in [out].
        """
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


def clean_io(inp_path: pathlib.Path, out_path: pathlib.Path):

    if os.path.isdir(inp_path) and os.path.isdir(out_path):
        return 'directory', inp_path, out_path

    mimetype = mimetypes.guess_type(inp_path)[0]
    if mimetype.startswith("video") and not mimetype == "video/gif":
        return 'video', inp_path, out_path.with_suffix(".mp")
    elif mimetype.startswith("image"):
        return 'image', inp_path, out_path.with_suffix(".mp")
    return None, inp_path, out_path.with_suffix(".mp")


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
    args = create_parser().parse_args()
    mime, in_path, out_path = clean_io(args.input, args.output)
    if mime is None:
        print("Invalid input")
        return
    
    extractor = setup(
        rank=args.rank,
        rho=args.rho,
        iterations=args.maxiter,
        descriptor_type=args.descriptor_type,
        keep_best=args.keep_best
    )

    if mime == 'video':
        grouped_stream = fm.from_video_grouped(in_path, args.group_size)
        extractor.save_from_video(
            grouped_stream,
            out_path,
            source_id=out_path.stem
        )
    elif mime == 'image':

        def _wrap(x):
            yield x

        img = fm.from_image(in_path)
        extractor.save(
            _wrap(img),
            out_path,
            source_id=out_path.stem
        )
    else:
        extractor.save_from_directory(
            in_path,
            out_path,
            group_size=args.group_size
        )


if __name__ == "__main__":
    main()
