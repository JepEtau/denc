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
    img_to_tensor,
)
import torch
from torch import Tensor


def main():
    cpu_count: int = int(3 * multiprocessing.cpu_count() / 4)
    device: str = 'cpu'

    # a list of images
    if sys.platform == 'linux':
        in_img_dir: str = absolute_path(f"~/z-personnel/mco/imgs/ep10_226_lr_j")
    else:
        in_img_dir: str = absolute_path(f"N:\\imgs\\ep01_215_upscale_j")

    in_img_fp: list[str] = sorted(
        [os.path.join(in_img_dir, f) for f in os.listdir(in_img_dir) if f.endswith(".png")]
    )
    in_img_fp = in_img_fp[:75]

    start_time = time.time()
    in_images = denc.load_images(
        filepaths=in_img_fp, cpu_count=cpu_count, dtype=np.float32
    )
    elapsed = time.time() - start_time
    print(f"[np.float32] loaded {len(in_images)} images in {1000 * (elapsed):.01f}ms ({len(in_images)/elapsed:.01f}fps) (cpu_count={cpu_count})")

    out_frames: list[np.ndarray] = list([
        img_to_tensor(d_img=torch.from_numpy(img), tensor_dtype=torch.float32, flip_r_b=True)
        for img in in_images
    ])


    # Write images as video
    print(lightcyan(f"h264"))
    out_video_fp: str = "h264_yuv420p_rec709_veryfast.mkv"
    out_media: MediaStream = denc.new(out_video_fp)
    vstream = out_media.video
    vstream.codec = VideoCodec.H264
    vstream.pix_fmt = PixFmt.YUV420P
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

    denc.write(out_media, frames=out_frames)


    print(lightcyan(f"h265"))
    out_media.video.filepath = "h265_yuv422p10_rec709_veryfast.mkv"
    vstream.codec = VideoCodec.H265
    vstream.pix_fmt = PixFmt.YUV422P10
    denc.write(out_media, frames=out_frames)




    print("Ended.")



if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()

