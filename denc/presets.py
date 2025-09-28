from dataclasses import dataclass


DEFAULT_H265_FFMPEG_ARGS: str = """
    -preset slow
    -crf 16
    -profile:v main422-10
    -x265-params sao=0
"""

DEFAULT_HVEC_NVENC_FFMPEG_ARGS: str = """
    -profile:v main
    -b_ref_mode disabled
    -tag:v hvc1
    -g 30
    -preset p7
    -tune hq
    -rc constqp
    -qp 17
    -rc-lookahead 20
    -spatial_aq 1
    -aq-strength 15
    -b:v 0
"""


@dataclass
class ColorSettings:
    colorspace: str | None = 'bt709'
    color_primaries: str | None = 'bt709'
    color_trc: str | None = 'bt709'
    color_range: str | None = 'tv'
