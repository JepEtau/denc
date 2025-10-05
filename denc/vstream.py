from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import math
import numpy as np
import os
import torch
from typing import Any, Literal, Optional, TYPE_CHECKING
from warnings import warn

from .colorpspace import (
    ColorRange,
    ColorSpace,
)
from .pxl_fmt import PIXEL_FORMATS
from .torch_tensor import np_to_torch_dtype
from .utils.path_utils import path_split
from .utils.p_print import *
from .utils.time_conversions import FrameRate
from .vcodec import (
    PixFmt,
    VideoCodec,
    vcodec_to_extension,
    CODEC_PROFILE,
)
if TYPE_CHECKING:
    from .media_stream import MediaStream


FShape = tuple[int, int, int]


class FieldOrder(Enum):
    PROGRESSIVE = 'progressive' # Progressive
    TOP_FIELD_FIRST = 'tt'      # Interlaced, top field coded and displayed first
    BOTTOM_FIELD_FIRST = 'bb'   # Interlaced, bottom field coded and displayed first
    TOP_FIELD_BOTTOM = 'tb'     # Interlaced, top coded first, bottom displayed first
    BOTTOM_FIELD_TOP = 'bt'     # Interlaced, bottom coded first, top displayed first



@dataclass(slots=True)
class PipeFormat:
    dtype: torch.dtype
    pix_fmt: Literal['rgb24', 'rgb48le']
    shape: FShape
    nbytes: int



@dataclass(slots=True)
class Device:
    name: str = 'cpu'
    dtype: torch.dtype = torch.float32



@dataclass(slots=True)
class DecoderResize:
    enabled: bool = False
    use_sar: bool = True
    algorithm: Literal['lanczos', 'bicubic'] = 'bicubic'
    side: Literal['width', 'height'] = 'width'



@dataclass
class VideoStream:
    filepath: str
    codec: VideoCodec | str

    shape: FShape

    # Raw Frame Rate
    #  how the muxer (container) or codec declares the frame timing,
    #  not necessarily the actual playback rate.
    #  may be inaccurate or higher than real frame rate,
    #  especially in variable frame rate (VFR) videos.
    frame_rate_r: FrameRate

    # Average Frame Rate
    #  computed from the video's total frames and total duration
    #  real playback: how many frames per second are actually shown
    frame_rate_avg: FrameRate

    frame_count: int
    duration: float

    pix_fmt: PixFmt

    is_frame_rate_fixed: bool = False

    sar: Fraction = Fraction(1, 1)
    dar: Fraction = Fraction(1, 1)

    is_interlaced: bool = False
    field_order: FieldOrder = FieldOrder.PROGRESSIVE

    profile: Optional[str] = ""

    color_range: Optional[ColorRange] = None
    color_space: Optional[ColorSpace] = None
    color_matrix: Optional[ColorSpace] = None
    color_primaries: Optional[ColorSpace] = None
    color_transfer: Optional[ColorSpace] = None

    metadata: Any = None

    resize: DecoderResize = field(default_factory=DecoderResize)
    device: Device = field(default_factory=Device)

    parent: Optional[MediaStream] = field(default=None, repr=False, compare=False)


    def __post_init__(self):
        pipe_pixel_format: dict = PIXEL_FORMATS[self.pix_fmt]['pipe_pxl_fmt']
        if pipe_pixel_format in (PixFmt.RGB24, PixFmt.RGBA24):
            pipe_dtype: torch.dtype = torch.uint8
        elif pipe_pixel_format in (PixFmt.RGB48, PixFmt.RGBA48):
            pipe_dtype: torch.dtype = torch.uint16
        else:
            raise NotImplementedError(f"not supported: {pipe_pixel_format}")

        shape = self._calculate_pipe_shape()
        self._pipe_format = PipeFormat(
            dtype=pipe_dtype,
            pix_fmt=pipe_pixel_format,
            shape=shape,
            nbytes=math.prod(shape) * torch.tensor([], dtype=pipe_dtype).element_size(),
            # device='cpu'
        )
        self._vcodec = self.codec


    def _calculate_pipe_shape(self) -> FShape:
        h, w, c = self.shape
        if self.resize.use_sar:
            if self.resize.side == 'width':
                w = int(w * self.sar.numerator / self.sar.denominator)
            else:
                h = int(h * self.sar.denominator / self.sar.numerator)
        if self.resize.enabled:
            raise ValueError(red("not yet implemented"))
        return (h, w, c)


    @property
    def pipe_format(self) -> PipeFormat:
        return self._pipe_format


    def set_pipe_format(self, dtype: torch.dtype | np.dtype) -> None:
        """Video pipe
        """
        if not isinstance(self, OutVideoStream):
            return
        torch_dtype: torch.dtype = np_to_torch_dtype(dtype)
        self._pipe_format.dtype = torch_dtype
        self._pipe_format.pix_fmt = 'rgb24' if torch_dtype == torch.uint8 else 'rgb48le'
        self._pipe_format.shape = self._calculate_pipe_shape()
        self._pipe_format.nbytes = (
            math.prod(self._pipe_format.shape)
            * torch.tensor([], dtype=torch_dtype).element_size()
        )


    def set_device(
        self,
        device: str,
        dtype: torch.dtype = torch.float32
    ) -> None:
        if device == "cpu":
            dtype = torch.float32
        self.device = Device(name=device, dtype=dtype)



# presets
class FFmpegPreset(Enum):
    DEFAULT = "medium"
    ULTRAFAST = "ultrafast"
    SUPERFAST = "superfast"
    VERYFAST = "veryfast"
    FASTER = "faster"
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    SLOWER = "slower"
    VERYSLOW = "veryslow"
    PLACEBO = "placebo"
_preset_keys = [preset.name for preset in FFmpegPreset]



@dataclass
class OutVideoStream(VideoStream):
    parent: Optional[MediaStream] = field(default=None, repr=False, compare=False)
    _extra_params: list[str] = field(default_factory=list)
    _preset: FFmpegPreset = FFmpegPreset.DEFAULT
    _crf: int = -1
    _profile: str = ""

    def __post_init__(self):
        super().__post_init__()
        self._codec: VideoCodec = self.codec
        self.set_pipe_format(dtype=self.dtype)

    @property
    def codec(self) -> VideoCodec:
        return self._codec


    @codec.setter
    def codec(self, codec: VideoCodec) -> None:
        self._codec = codec
        if self.filepath:
            directory, basename, _ = path_split(self.filepath)
            self.filepath = os.path.join(
                directory, f"{basename}{vcodec_to_extension[codec]}"
            )
            if self.parent is not None:
                self.parent.filepath = self.filepath


    @property
    def extra_params(self) -> list[str]:
        return self._extra_params


    @extra_params.setter
    def extra_params(self, params: str | list[str]) -> None:
        if isinstance(params, str):
            self._extra_params = [x for x in params.split(" ") if x]
        elif isinstance(params, list | tuple):
            self._extra_params = [x for x in params if x]
        else:
            raise TypeError(f"Wrong type: {type(params)}")


    @property
    def preset(self) -> FFmpegPreset:
        return self._preset


    @preset.setter
    def preset(self, preset: FFmpegPreset) -> None:
        if preset.name in _preset_keys:
            self._preset = preset


    @property
    def crf(self) -> int:
        # libx264	    0–51	(23)
        # libx265	    0–51	(28)
        # libvpx-vp9	0–63	(31)
        if self.codec == VideoCodec.VP9:
            return max(-1, min(self._crf, 63))
        return max(-1, min(self._crf, 51))


    @crf.setter
    def crf(self, crf: int) -> None:
        self._crf = max(0, min(crf, 63))


    @property
    def profile(self) -> str:
        if self._profile:
            if self._profile in CODEC_PROFILE[self._vcodec]['available']:
                return self._profile
            else:
                warn(f"\'{self._profile}\' is not a valid profile for {self._vcodec}, available: {CODEC_PROFILE[self._vcodec]['available']}")
                return ""
        else:
            if (
                not self._profile and CODEC_PROFILE[self._vcodec]['default']
            ):
                return CODEC_PROFILE[self._vcodec]['default']
        return self._profile


    @profile.setter
    def profile(self, profile: str) -> None:
        if not profile in CODEC_PROFILE[self._vcodec]['available']:
            warn(f"\'{self._profile}\' is not a valid profile for {self._vcodec}, available: {CODEC_PROFILE[self._vcodec]['available']}")
        self._profile = profile
