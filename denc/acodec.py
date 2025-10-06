from enum import Enum


class AudioCodec(Enum):
    AAC = "aac"
    MP3 = "libmp3lame"
    OPUS = "libopus"


acodec_to_ffmpeg_acodec: dict[AudioCodec, str] = {
    AudioCodec.AAC: "aac",
    AudioCodec.MP3: "libmp3lame",
    AudioCodec.OPUS: "libopus",
}
