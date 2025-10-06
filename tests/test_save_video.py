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
from denc import (
    MediaStream,
    PIXEL_FORMATS,
    VideoCodec,
    PixFmt,
    ColorSpace,
    FFmpegPreset,
)
import torch
from torch import Tensor


def main():
    cpu_count: int = int(3 * multiprocessing.cpu_count() / 4)
    device: str = 'cpu'

    # a list of images
    in_img_dir: str = absolute_path(f"~/z-personnel/mco/imgs/ep10_226_lr_j")
    in_img_fp: list[str] = sorted(
        [os.path.join(in_img_dir, f) for f in os.listdir(in_img_dir) if f.endswith(".png")]
    )
    in_img_fp = in_img_fp[:75]

    start_time = time.time()
    in_frames = denc.load_images(
        filepaths=in_img_fp, cpu_count=cpu_count, dtype=np.float32
    )
    elapsed = time.time() - start_time
    print(f"[np.float32] loaded {len(in_frames)} images in {1000 * (elapsed):.01f}ms ({len(in_frames)/elapsed:.01f}fps) (cpu_count={cpu_count})")

    # out_frames: list[Tensor] = list(
    #     [img_to_tensor(img, tensor_dtype=torch.float32) for img in in_images]
    # )

    # Write images as video
    out_video_fp: str = "ep10_226_lr_j.mkv"
    out_media: MediaStream = denc.new(out_video_fp)
    vstream = out_media.video
    vstream.codec = VideoCodec.H264
    vstream.pix_fmt = PixFmt.YUV422P10
    vstream.color_space = ColorSpace.REC709
    # vstream.crf = 20
    vstream.preset = FFmpegPreset.VERYFAST
    # vstream.profile = "main"
    vstream.set_device(device=device, dtype=torch.float32)

    # pipe_format is not valid yet because it depends on the tensor
    # could be set using vstream.set_pipe_format or wait for the encoder to set it
    print(vstream.pipe_format)
    vstream.set_pipe_format(torch.uint8)
    print(vstream.pipe_format)
    print(vstream.codec)

    denc.write(out_media, frames=in_frames)

    print("Ended.")



if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()

