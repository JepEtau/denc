from cgitb import grey
from enum import Enum
import re
import subprocess

from .utils.p_print import *
from .utils.tools import ffmpeg_exe

class PixFmt(Enum):
    YUV420P = "yuv420p"
    YUV422P10 = "yuv422p10le"
    RGB24 = "rgb24"
    RGB48 = "rgb48"


def list_pixel_formats() -> dict[str, dict[str, bool | int | str]]:
    pixel_formats: dict[str, dict[str, bool | int | str]] = {}


    ffmpeg_command = [ffmpeg_exe, "-hide_banner", "-pix_fmts"]
    try:
        process = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE)
        ffmpeg_pixl_fmts: str = process.stdout.decode('utf-8')
    except:
        raise ValueError(red(f"Failed to get supported pixel formats"))


    _pixel_formats: list[str] | None = None
    if (
        _pixel_formats := re.findall(
            re.compile(r"[IOHBP.]{5}\s+([a-z_\d]+)\s+([1234]{1})\s+(\d+)\s+([\d-]+)"),
            ffmpeg_pixl_fmts
        )
    ):
        for f in _pixel_formats:
            k, nc, bpp, bit_depths = f
            c_order: str = ''
            if 'rgb' in k:
                c_order = 'rgb'
            elif 'gbr' in k:
                c_order = 'gbr'
            elif 'bgr' in k:
                c_order = 'bgr'
            elif 'yuv' in k:
                c_order = 'yuv'
            elif 'gray' in k:
                c_order = 'gray'

            if (
                'be' in k
                or 'j' in k
                or 'x' in k
            ):
                # not supported
                c_order = ''

            pixel_formats[k] = {
                'c': int(nc),
                'bpp': int(bpp),
                'pipe_bpp': int(nc) * max(list(map(int, bit_depths.split('-')))),
                'c_order': c_order,
                'supported': True if int(nc) in (1, 3) and c_order != '' else False,
            }
    else:
        raise ValueError(red("Failed extracting pixel format"))

    return pixel_formats


PIXEL_FORMATS = list_pixel_formats()


# Debug
if False:
    for k, v in PIXEL_FORMATS.items():
        if v['supported']:
            print(lightgreen(f"{k:<10}"), f"\tc_order={v['c_order']}, nc={v['c']}, bpp={v['bpp']}, pipe_bpp={v['pipe_bpp']}")
        else:
            print(darkgrey(f"{k:<10}\tnc={v['c']}, bpp={v['bpp']}, pipe_bpp={v['pipe_bpp']}"))
