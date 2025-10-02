import multiprocessing
import os
from pprint import pprint
import signal
import sys
import time

import numpy as np
import denc
from denc.utils.p_print import lightcyan
from denc.utils.path_utils import absolute_path
from denc import MediaStream



def main():

    if sys.platform == 'win32':
        in_video_dir: str = os.path.join(__file__, os.pardir, os.pardir, os.pardir, "benchmark")
        in_video_filename: str = "smpte_h264_640x480_yuv420p_bt709_limited_50.mkv"

    elif sys.platform == 'linux':
        in_video_dir: str = f"~/mco/cache/ep10/sim_lr"
        in_video_filename: str = "ep10_episode_213__j_sim_lr_original.mkv"

    in_video_dir = absolute_path(in_video_dir)

    in_videos: list[str] = sorted(
        [
            os.path.join(in_video_dir, f)
            for f in os.listdir(in_video_dir)
            if f.endswith(".mkv") or f.endswith(".mxf")
        ]
    )

    for f in in_videos:
        in_video_fp = os.path.join(in_video_dir, f)
        print(lightcyan(f"Input video file:"), f"{in_video_fp}")
        if not os.path.isfile(in_video_fp):
            raise FileExistsError(f"Missing file.")

        try:
            media: MediaStream = denc.open(in_video_fp)
            print(lightcyan(media.video.pipe_format))
            pprint(media.video)
        except:
            pass

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()

