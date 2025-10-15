from __future__ import annotations
import math
from pprint import pprint
from queue import Queue
from threading import Event
import numpy as np
import torch
from torch import Tensor
from torch.cuda import StreamContext

from .base_thread import BaseThread, NnFrame
from .encoder import encoder_subprocess
from .media_stream import MediaStream
from .vstream import FShape, PipeFormat, VideoStream

from .dh_transfers import dtoh_transfer
from .torch_tensor import tensor_to_img



class EncoderThread(BaseThread):
    def __init__(
        self,
        media_stream: MediaStream,
        name: str | None = None,
        use_predefined_settings: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(name=name, verbose=verbose)
        self._encoded: int = 0
        self._stop_event: Event = Event()
        self.in_queue: Queue = Queue(1)
        self.media_stream = media_stream
        # Start the encoder by using settings defined in the media stream
        if use_predefined_settings:
            self.e_subprocess = encoder_subprocess(vstream=media_stream.video)
        else:
            self.e_subprocess = None

        self.progress_thread = None


    @property
    def encoded(self) -> int:
        return self._encoded


    @torch.inference_mode()
    def run(self) -> None:
        verbose: bool = self.verbose

        vstream: VideoStream = self.media_stream.video
        pipe: PipeFormat = vstream.pipe_format

        # Flow control
        in_queue: Queue = self.in_queue
        received: int = 0
        remaining: int = self.media_stream.seek.count

        # Wait for first frame to start encoder
        if self.e_subprocess is None:
            input = in_queue.get(block=True)
            if input is None or self._stop_event.is_set():
                if verbose:
                    print(purple("[V][E] Received Null tensor"))
                self._processing = False
                return

            frame: NnFrame = input
            d_tensor = frame.tensor
            _, c, h, w = d_tensor.shape
            pipe.shape = (h, w, c)
            pipe.nbytes = (
                math.prod(pipe.shape) * torch.tensor([], dtype=pipe.dtype).element_size()
            )
            cuda = bool("cuda" in str(d_tensor.device))
            self.e_subprocess = encoder_subprocess(vstream=vstream)
            skip_first_frame = True

        else:
            cuda = bool("cuda" in vstream.device)

        # Output stream
        img_dtype: np.dtype = pipe.dtype
        img_shape: FShape = pipe.shape

        # Create a cuda stream and allocate Host memory
        if cuda:
            cuda_stream: torch.cuda.Stream = torch.cuda.Stream(vstream.device)
            stream_context: StreamContext = torch.cuda.stream(cuda_stream)
            host_mem: Tensor = torch.empty(
                img_shape, dtype=pipe.dtype, pin_memory=True
            )
        else:
            from contextlib import nullcontext
            cuda_stream = nullcontext()


        with stream_context:
            while (
                not self._stop_event.is_set()
                and remaining > 0
            ):
                if skip_first_frame:
                    skip_first_frame = False
                else:
                    # Wait for a frame or a poison pill
                    input = in_queue.get(block=True)
                    if input is None or self._stop_event.is_set():
                        if verbose:
                            print(purple("[V][E] Received Null tensor"))
                        self.end_encoding()
                        break

                frame: NnFrame = input
                d_tensor = frame.tensor

                # Get tensor from frame
                if verbose:
                    print(
                        purple(f"[V][E] Received no. {received}:"),
                        f"{d_tensor.shape}, {d_tensor.dtype}, {d_tensor.data_ptr()}"
                    )
                    received += 1

                d_img: Tensor = tensor_to_img(tensor=d_tensor, img_dtype=img_dtype)

                out_img: np.ndarray
                if cuda:
                    out_img = dtoh_transfer(
                        host_mem=host_mem,
                        d_img=d_img,
                        cuda_stream=cuda_stream
                    )

                else:
                    out_img = d_img.contiguous().numpy()

                frame.tensor = None

                if verbose:
                    print(
                        purple(f"[V][E] send to pipe:"),
                        f"{out_img.shape}, {out_img.dtype}"
                    )

                try:
                    self.e_subprocess.stdin.write(out_img)
                except:
                    print(f"failed send {type(out_img)}")
                    stdout: str = self.e_subprocess.stdout.read().decode("utf-8")
                    pprint(stdout.split('\n'))
                    break

                remaining -= 1
                sent = 1

                del frame.tensor
                if self.progress_thread is not None:
                    self.progress_thread.put(sent)
                self._encoded += sent

                if self.producer is not None:
                    self.producer.set_produce_flag()

        #     print(red(f"[V][E] Error while executing: "), " ".join(encoder_command))
        self._processing = False
        if verbose:
            print(
                purple(f"[V][E] All frames encoded or error"),
                f"{self._encoded}",
                flush=True
            )
        self.end_encoding()


    def end_encoding(self) -> bool:
        # Close output video
        if self.e_subprocess is None:
            return

        stdout_bytes: bytes | None = None
        try:
            # Arbitrary timeout value
            stdout_bytes, _ = self.e_subprocess.communicate(
                input='', timeout=60
            )
        except:
            self.e_subprocess.kill()
            pass

        if stdout_bytes is not None:
            std_str = stdout_bytes.decode('utf-8)')
            # pprint(std_str)
            # TODO: parse the output file ?
            for l in std_str.split("\n"):
                if (
                    not l.startswith("x265 [info]")
                    and not l.startswith("frame=")
                ):
                    print(l.strip())
        return True


    def stop(self, force: bool=False) -> None:
        self._stop_event.set()
        if force:
            while not self.in_queue.empty():
                self.in_queue.get_nowait()
        self.put_frame(None)
        self._processing = False


    def put_frame(self, frame: NnFrame) -> bool:
        if self._processing:
            self.in_queue.put(frame)
        return True

