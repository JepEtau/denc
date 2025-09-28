
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Optional

from .p_print import *


if TYPE_CHECKING:
    from .vstream import VideoStream

from .time_conversions import (
    FrameRate,
    frame_to_sexagesimal,
    sexagesimal_to_frame,
    frame_to_s
)


@dataclass
class Seek:
    vstream: VideoStream
    _start: Optional[int] = -1
    _count: Optional[int] = -1
    _to: Optional[int] = -1

    @property
    def start(self) -> int:
        if self._start < 0:
            if self._to > 0 and self._count > 0:
                return self._to - self._count
            return 0
        return self._start


    @property
    def start_hms(self) -> str:
        if self.start > 0:
            return frame_to_sexagesimal(self.start, self.vstream.frame_rate_r)
        return ""


    @start.setter
    def start(self, start: int | str) -> None:
        if isinstance(start, str) and start:
            # Convert from str to frame no.
            if result := re.search(re.compile(r"^(\d+)f$"), start):
                start = int(result.group(1))
            else:
                start = sexagesimal_to_frame(start, self.vstream.frame_rate_r)
        if start >= self.vstream.frame_count:
            raise ValueError(red(f"Erroneous start value: {self.start} > {self.vstream.frame_count}"))
        self._start = start


    @property
    def count(self) -> int:
        return min(self.vstream.frame_count - self._start, self._count)


    @property
    def duration(self) -> str:
        if self._count <= 0:
            if self._start > 0 and self._to > self._start:
                return min(self._to - self._start, self.vstream.frame_count - self._start)
            return ""
        return f"{frame_to_s(self.count, self.vstream.frame_rate_r)}"


    @count.setter
    def count(self, count: int | str) -> None:
        if isinstance(count, str) and count:
            # Convert from str to frame no.
            if result := re.search(re.compile(r"^(\d+)f$"), count):
                count = int(result.group(1))
            else:
                count = sexagesimal_to_frame(count, self.vstream.frame_rate_r)
        self._count = count
        self._to = -1



    @property
    def to(self) -> int:
        if self._to <= self._start or self._to > self._start + self._count:
            raise ValueError(red(f"Erroneous end value: {self._to} < {self._start}"))
        return self._to


    @property
    def to_hms(self) -> str:
        if self._to <= 0:
            return ""
        return frame_to_sexagesimal(self._to, self.vstream.frame_rate_r)


    @to.setter
    def to(self, to: int | str) -> None:
        if isinstance(to, str) and to:
            # Convert from str to frame no.
            if result := re.search(re.compile(r"^(\d+)f$"), to):
                to = int(result.group(1))
            else:
                to = sexagesimal_to_frame(to, self.vstream.frame_rate_r)
        if to <= self._start:
            raise ValueError(red(f"Erroneous end value: {to} < {self._start}"))
        self._to = to

        if self._start > 0:
            self._count = self._to - self._start
        else:
            self._count = self._to




    def __str__(self):
        string: str = f"""Seek(
    _start={self._start},
    start_hms={self.start_hms},
    _count={self._count},
    _to={self._to},

    duration={self.duration},
    vstream.framecount={self.vstream.frame_count}

    start={self.start},
    count={self.count},
)"""
        return string # .replace("\n", "").replace("\r", "").replace(" ", "")




    def _consolidate(
        vstream: VideoStream,
        start: int | str = 0,
        duration: int | str = -1,
        end: int | str = -1,
    ) -> Seek:
        # vstream: VideoStream = stream.video

        if not vstream.is_frame_rate_fixed:
            raise NotImplementedError("[E] variable frame rate is not supported yet")

        frame_rate: FrameRate = vstream.frame_rate_r


        # Start
        start_no: int = 0
        if isinstance(start, str) and start:
            # Convert from str to frame no.
            if result := re.search(re.compile(r"^(\d+)f$"), start):
                start_no = int(result.group(1))
            else:
                start_no = sexagesimal_to_frame(start, frame_rate)
        else:
            start_no = start
        if start_no >= vstream.frame_count:
            raise ValueError(red(f"Erroneous seek start: {start} >= {vstream.frame_count}"))
        print(yellow(f"start:"))
        # Duration
        count: int = -1
        if isinstance(duration, str) and duration:
            if result := re.search(re.compile(r"^(\d+)f$"), duration):
                count = int(result.group(1))
            else:
                count = sexagesimal_to_frame(duration, frame_rate)
        else:
            count = duration
        count = min(count, vstream.frame_count)

        # End
        end_no: int = -1
        if isinstance(end, str) and end:
            if result := re.search(re.compile(r"^(\d+)f$"), end):
                end_no = int(result.group(1))
            else:
                end_no = sexagesimal_to_frame(end, frame_rate)
        else:
            end_no = end
        count = min(end_no, vstream.frame_count)

        max_count = vstream.frame_count
        if start_no > 0:
            if end_no >= start_no:
                raise ValueError(red(f"Erroneous seek end: {end} > {start}"))
            max_count = end_no - start_no

        # Only start is specified
        if count <= 0 and end_no <= 0:
            count = max_count - start_no
        # count is specified and has the priority
        elif count > 0:
            end_no = start_no + count
        # end is specified: low priority
        elif end_no > 0:
            if end_no < start_no:
                raise ValueError(red(f"Erroneous seek end: {end} < {start}"))
            count = end_no - start_no

        return Seek(
            start=start_no,
            start_hms=frame_to_sexagesimal(start_no, frame_rate),
            count=count,
            duration=frame_to_sexagesimal(count, frame_rate) if count else "",
            to=end_no,
            to_hms=frame_to_sexagesimal(end_no, frame_rate) if end_no else "",
        )
