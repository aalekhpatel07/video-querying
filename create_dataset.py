import pathlib
import os
import mimetypes
import argparse
import typing as tp
from core.perturb import image as pim
from core.perturb import video as piv


def perturb_video(path: pathlib.Path, **kwargs):
    print(path, kwargs)
    return 0


def perturb_image(path: pathlib.Path, **kwargs):
    print(path, kwargs)
    return 0


def read_parser():
    parser = argparse.ArgumentParser(description="Create a dataset of media.")
    parser.add_argument(
        "-s",
        "--source",
        type=pathlib.Path,
        help="The source directory that contains all the media.",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=pathlib.Path,
        help="The directory for storing the modified media in.",
        required=True,
    )

    return parser.parse_args()


def media_files(
    directory: pathlib.Path,
) -> tp.Generator[tp.Tuple[str, pathlib.Path], None, None]:
    for root, dirs, files in os.walk(directory):
        for file in files:
            mime_type = mimetypes.guess_type(os.path.join(root, file))[0]
            if mime_type.startswith("video"):
                yield "video", os.path.join(root, file)
            elif mime_type.startswith("image"):
                yield "image", os.path.join(root, file)
        for d in dirs:
            yield from media_files(pathlib.Path(os.path.join(root, d)))


def main():
    args = read_parser()
    source, dest = args.source, args.output_dir
    for mime, path in media_files(source):
        transformed = None
        if mime == "image":
            transformed = perturb_image(path, **args.__dict__)
        elif mime == "video":
            transformed = perturb_video(path, **args.__dict__)
        break
    pass


if __name__ == "__main__":
    main()
