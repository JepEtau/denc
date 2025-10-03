import multiprocessing
import os
from pprint import pprint
import re
import signal
import sys
import time

import numpy as np
import denc
from denc.utils.p_print import *
from denc.utils.path_utils import absolute_path
from denc import MediaStream



def main():

    in_video_dir: str = absolute_path(
        os.path.join(__file__, os.pardir, os.pardir, os.pardir, "benchmark")
    )

    in_videos: list[str] = sorted(
        [
            f
            for f in os.listdir(in_video_dir)
            # if f.endswith(".mkv") or f.endswith(".mxf")
            # if "smpte" in f
        ]
    )

    filename_pattern = re.compile(r"""
    ^([^_]+)            # codec
    _(\d+)x(\d+)        # resolution width x height
    _([a-z0-9]+)        # pixel format
    (?:_([a-z0-9]+))?   # colorspace
    (?:_([a-z0-9]+))?   # (pal / ntsc / none)
    _(full|limited)     # range
    _([a-z0-9]+)        # pattern
    \.                 # extension
    """, re.VERBOSE)

    for f in in_videos:
        in_video_fp = os.path.join(in_video_dir, f)
        print(lightcyan(f"{f}"), end='')
        if not os.path.isfile(in_video_fp):
            raise FileExistsError(f"Missing file.")

        try:
            media: MediaStream = denc.open(in_video_fp)
            print(f"\t{media.video.pipe_format}", end='\t')
            # pprint(media.video)
        except Exception as e:
            print(red(f"\n\tnot supported"))
            # media: MediaStream = denc.open(in_video_fp)
            continue
        print()

        if result := re.search(filename_pattern, f):
            codec, width, height, pix_fmt, color_space, ntsc_pal, color_range, video_pattern = result.groups()
            print(result.groups())
        else:
            print(red("  not recognized"))
       


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()

