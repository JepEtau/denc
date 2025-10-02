import multiprocessing
import os
from pprint import pprint
import signal
import time

import numpy as np
import denc
from denc.utils.p_print import lightcyan
from denc.utils.path_utils import absolute_path
from denc import MediaStream



def main():
    in_video_fp: str = f"~/mco/cache/ep10/sim_lr/ep10_episode_213__j_sim_lr_original.mkv"

    in_video_fp = absolute_path(in_video_fp)
    print(lightcyan(f"Input video file:"), f"{in_video_fp}")
    if not os.path.isfile(in_video_fp):
        raise FileExistsError(f"Missing file.")

    media: MediaStream = denc.open(in_video_fp)


    pprint(media.video)
    # print(media.seek)
    # media.video.set_pipe_format(dtype=np.uint8)    
    # print(media.video.pipe_format)

    # media.video.set_pipe_format(dtype=np.uint16)    
    # print(media.video.pipe_format)

    # media.video.set_pipe_format(dtype=np.float32)    
    print(media.video.pipe_format)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()

