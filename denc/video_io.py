from __future__ import annotations
from fractions import Fraction
import os
import numpy as np

from .media_stream import (
    AudioInfo,
    MediaStream,
    SubtitleInfo,
    VideoStream,
    probe_media_file,
)
from .utils.p_print import *
from .utils.path_utils import absolute_path
from .pxl_fmt import PIXEL_FORMATS, PixFmt
from .utils.time_conversions import FrameRate
from .vcodec import VideoCodec
from .vstream import FieldOrder, OutVideoStream



def open(filepath: str, verbose: bool = False) -> MediaStream:
    in_video_fp: str = absolute_path(filepath)
    if verbose:
        print(lightcyan(f"Input video file:"), f"{in_video_fp}")
    if not os.path.isfile(in_video_fp):
        raise ValueError(red(f"Error: missing input file {in_video_fp}"))

    media_info = probe_media_file(in_video_fp)
    duration_s = float(media_info['format']['duration'])

    # Use the first video track
    v_stream: dict[str, str] = [
        stream for stream in media_info['streams'] if stream['codec_type'] == 'video'
    ][0]
    audio_info: AudioInfo = AudioInfo(
        nstreams=len([
            stream for stream in media_info['streams'] if stream['codec_type'] == 'audio'
        ])
    )
    subs_info: SubtitleInfo = SubtitleInfo(
        nstreams=len([
            stream for stream in media_info['streams'] if stream['codec_type'] == 'subtitle'
        ])
    )

    # Video stream
    # Only first stream is used
    # Determine nb of channels and bpp
    is_supported: bool = False
    pix_fmt = v_stream.get('pix_fmt', None)
    try:
        v = PIXEL_FORMATS[pix_fmt]
        is_supported = v['supported']
        shape = (v_stream['height'], v_stream['width'], v['c'])
    except:
        print(pix_fmt)
        raise
        pass
    if not is_supported:
        raise ValueError(f"{pix_fmt} is not supported")


    field_order = FieldOrder._value2member_map_[v_stream.get('field_order', 'progressive')]

    video_info: VideoStream = VideoStream(
        is_input=True,
        filepath=in_video_fp,
        shape=shape,
        sar=Fraction(v_stream.get('sample_aspect_ratio', '1:1').replace(':', '/')),
        dar=Fraction(v_stream.get('display_aspect_ratio', '1:1').replace(':', '/')),

        field_order=field_order,

        frame_rate_r=Fraction(v_stream.get('r_frame_rate', '0')),
        frame_rate_avg=Fraction(v_stream.get('avg_frame_rate', '0')),

        codec=v_stream['codec_name'],
        pix_fmt=pix_fmt,
        # Colors
        color_space=v_stream.get('color_space', None),
        color_matrix=v_stream.get('color_matrix', None),
        color_transfer=v_stream.get('color_transfer', None),
        color_primaries=v_stream.get('color_primaries', None),
        color_range=v_stream.get('color_range', None),

        duration=duration_s,
        metadata=v_stream.get('tags', None),

        dtype=np.uint8,
        bpp=24,
        c_order='rgb',
        frame_count=0,
    )

    tags_to_discard: tuple[str, ...]
    if isinstance(video_info.metadata, dict):
        tags_to_discard = (
            'duration',
            'encoder',
            'creation_time',
            'handler_name',
            'vendor_id',
            'file_package_umid'
        )
        tag_name: str
        for tag_name in list(video_info.metadata.keys()).copy():
            if tag_name.lower() in tags_to_discard:
                try:
                    del video_info.metadata[tag_name]
                except:
                    pass

    # Tags for DNxHD / DNxHR are stored in format struct
    if video_info.codec == VideoCodec.DNXHR:
        tags_to_discard = (
            'application_platform',
            'company_name',
            'generation_uid',
            'material_package_umid',
            'operational_pattern_ul',
            'product_name',
            'product_uid',
            'product_version',
            'product_version_num',
            'timecode',
            'toolkit_version_num',
            'uid',
        )

        tags: dict[str, str]
        if tags := media_info['format']['tags']:
            tag_name: str
            for tag_name in list(tags.keys()).copy():
                if tag_name.lower() in tags_to_discard:
                    try:
                        del video_info.metadata[tag_name]
                    except:
                        pass
                    continue
                video_info.metadata[tag_name] = tags[tag_name]

    # Is interlaced?
    if fo := v_stream.get('field_order', None):
        video_info.is_interlaced = bool(fo != FieldOrder.PROGRESSIVE.value)
    video_info.is_frame_rate_fixed = bool(video_info.frame_rate_r == video_info.frame_rate_avg)


    c_order = v['c_order']
    if c_order not in ['rgb', 'bgr', 'yuv']:
        raise ValueError(f"{v['c_order']} is not supported")
    video_info.c_order = c_order


    video_info.bpp = v['bpp']
    video_info.frame_count = int(
        (video_info.duration * video_info.frame_rate_r) + 0.5
    )

    tags = v_stream.get('tags', None)
    if tags is not None:
        tag_frame_count = tags.get('NUMBER_OF_FRAMES', '')
        if tag_frame_count is not None:
            tag_frame_count: int = int(tag_frame_count)
            if video_info.frame_count != tag_frame_count:
                video_info.frame_count = tag_frame_count
                video_info.frame_rate_r = Fraction(tag_frame_count, video_info.duration)
                video_info.frame_rate_avg = video_info.frame_rate_r

    return MediaStream(
        filepath=filepath,
        video=video_info,
        audio=audio_info,
        subtitles=subs_info
    )



def new(filepath: str, preset = None) -> MediaStream:
    vstream = OutVideoStream(
        filepath=filepath,
        codec=VideoCodec.H265,
        shape=(0, 0, 0),
        dtype=np.uint16,
        bpp=10,
        c_order='rgb',
        frame_rate_r=FrameRate(25, 1),
        frame_rate_avg=FrameRate(25, 1),
        frame_count=0,
        duration=0,
        pix_fmt=PixFmt.YUV422P10,
    )
    mi: MediaStream=MediaStream(
        filepath=filepath,
        video=vstream,
        audio=AudioInfo(nstreams=0),
        subtitles=SubtitleInfo(nstreams=0),
    )
    vstream.parent = mi

    return mi
