from __future__ import annotations
from dataclasses import dataclass, field
import json
import subprocess

from .encoder import write
from .seek import Seek
from .vstream import OutVideoStream, VideoStream


@dataclass
class AudioInfo:
    nstreams: int = 0


@dataclass
class SubtitleInfo:
    nstreams: int = 0



@dataclass
class MediaStream:
    filepath: str
    video: VideoStream | OutVideoStream
    audio: AudioInfo | None = None
    subtitles: SubtitleInfo | None = None
    seek: Seek = field(init=False)


    def __post_init__(self):
        self.video.parent = self
        self.seek = Seek(vstream=self.video)


    def set_seek(self, start: int = -1, count: int = -1, end: int = -1) -> None:
        def _is_set(arg) -> bool:
            if (
                (isinstance(arg, int) and arg >= 0)
                or (isinstance(arg, str) and arg)
            ):
                return True
            return False

        if _is_set(start):
            self.seek.start = start
        if _is_set(count):
            self.seek.count = count
        if _is_set(end):
            self.seek.to = end


    def write(self, frames):
        return write(self, frames)



