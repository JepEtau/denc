from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import TypedDict

from .pxl_fmt import PixFmt


class PatternName(Enum):
    SMPTEHDBARS = "smptehdbars"
    YUVTESTSRC = "yuvtestsrc"
    COLORCHART = "colorchart"


pretty_pattern_name: dict[PatternName, str] = {
    PatternName.SMPTEHDBARS: "smpte",
    PatternName.YUVTESTSRC: "yuv",
    PatternName.COLORCHART: "cchart",
}


class VideoCodec(Enum):
    H264 = "h264"
    H265 = "h265"
    FFV1 = "FFv1"
    DNXHR = "DNxHR"
    VP9 = "VP9"
    PRORES = "ProRes"
    H264_NVENC = "h264_nvenc"
    HEVC_NVENC = "hevc_nvenc"


vcodec_to_ffmpeg_vcodec: dict[VideoCodec, str] = {
    VideoCodec.H264: "libx264",
    VideoCodec.H264_NVENC: "h264_nvenc",
    VideoCodec.H265: "libx265",
    VideoCodec.H265: "libx265",
    VideoCodec.VP9: "libvpx-vp9",
    VideoCodec.FFV1: "ffv1",
    VideoCodec.DNXHR: "dnxhd",
    VideoCodec.PRORES: "prores_ks",
}


class ProResProfile(IntEnum):
    Proxy = 0
    LT = 1
    Standard = 2    # ProRes 422
    HQ = 3
    HQ_ALPHA = 4


# default profile
vcodec_default_profile: dict[VideoCodec, str] = {
    VideoCodec.H264: "",
    VideoCodec.H265: "",
    VideoCodec.VP9: "",
    VideoCodec.FFV1: "",
    VideoCodec.DNXHR: "dnxhr_hqx",
    VideoCodec.PRORES: str(ProResProfile.Standard),
}


vcodec_to_extension: dict[VideoCodec, str] = {
    VideoCodec.H264: ".mkv",
    VideoCodec.H265: ".mkv",
    VideoCodec.FFV1: ".mkv",
    VideoCodec.VP9: ".mkv",
    VideoCodec.DNXHR: ".mxf",
    VideoCodec.PRORES: ".mov",
}





pixfmts: dict[VideoCodec, tuple[PixFmt]] = {
    VideoCodec.H264: (PixFmt.YUV420P,),
    VideoCodec.H265: (PixFmt.YUV420P, PixFmt.YUV422P10),
    VideoCodec.FFV1: (PixFmt.RGB24, PixFmt.RGB48),
    VideoCodec.DNXHR: (PixFmt.YUV422P10,),
    VideoCodec.VP9: (PixFmt.YUV420P, PixFmt.YUV422P10),
    VideoCodec.PRORES: (PixFmt.YUV422P10,),
}

supported_pixfmt: dict[VideoCodec, tuple[PixFmt]] = {
    VideoCodec.H264: (PixFmt.YUV420P,),
    VideoCodec.H265: (PixFmt.YUV420P, PixFmt.YUV422P10),
    VideoCodec.FFV1: (PixFmt.YUV420P, PixFmt.YUV422P10, PixFmt.RGB24, PixFmt.RGB48),
    VideoCodec.DNXHR: (PixFmt.YUV420P, PixFmt.YUV422P10),
    VideoCodec.VP9: (PixFmt.YUV420P, PixFmt.YUV422P10),
}




@dataclass
class FFv1Settings:
    level: int = 1
    coder: int = 1
    context: int = 1
    g: int = 1
    threads: int = 8


class CodecProfile(TypedDict):
    available: list[str]
    default: str


codec_profile: dict[VideoCodec, CodecProfile] = {
    VideoCodec.H264: CodecProfile(
        available=("baseline", "main", "high", "high10", "high422", "high444"),
        default=""
    ),
    VideoCodec.H265: CodecProfile(
        available=("main", "main10", "mainstillpicture"), default=""
    ),
    # mpeg2video    # "simple, main, high",
    VideoCodec.VP9: CodecProfile(available=(), default=""), # "No standard profiles exposed (VP8/VP9 use levels instead)"
    # libaom-av1    # "main, high, professional",
    VideoCodec.H264_NVENC: CodecProfile(
        available=("baseline", "main", "high", "high444"), default=""
    ),
    VideoCodec.HEVC_NVENC: CodecProfile(available=("main", "main10", "rext"), default=""),
    VideoCodec.DNXHR: CodecProfile(
        available=("dnxhr_hqx", "lb", "sq", "hq", "hqx", "444"),
        default="dnxhr_hqx"
    ),
    VideoCodec.PRORES: CodecProfile(
        available=("proxy", "lt", "standard", "hq", "4444", "4444xq"),
        default=str(ProResProfile.Standard)
    ),
}


